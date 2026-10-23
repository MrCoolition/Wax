import { createHash } from "node:crypto";
import type { Pool, PoolClient } from "pg";
import { z } from "zod";
import type {
  CatalogCandidate,
  Collector,
  CopyFields,
  CreateRecordInput,
  Crate,
  LibraryData,
  RecordCopy,
  SharedRecord,
  UpdateRecordInput,
  WizardChange,
} from "@/lib/contracts";
import { CONDITIONS } from "@/lib/contracts";
import { getPool, withTransaction } from "./db";
import { HttpError } from "./http";
import { catalogArtworkPath, fetchRelease } from "./catalog";

type Client = Pool | PoolClient;
const lockCollection = (client: PoolClient, userId: string) =>
  client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [
    userId,
  ]);
const uuid = z.string().uuid();
const shortText = z.string().trim().max(500);
const condition = z.enum(CONDITIONS);
export const candidateSchema = z
  .object({
    discogsId: z.number().int().positive().nullable(),
    artist: shortText.min(1),
    title: shortText.min(1),
    year: z.number().int().min(1850).max(2200).nullable(),
    genres: z.array(z.string().trim().max(100)).max(30),
    label: shortText,
    catalogNumber: shortText,
    country: shortText,
    format: shortText,
    coverUrl: z
      .union([
        z.string().url(),
        z
          .string()
          .max(64)
          .regex(/^\/api\/catalog\/[1-9]\d*\/artwork$/),
      ])
      .nullable(),
    sourceUrl: z.string().url().nullable(),
  })
  .strict();
const copyShape = {
  mediaCondition: condition,
  sleeveCondition: condition,
  purchasePrice: z
    .string()
    .regex(/^\d{1,10}(\.\d{1,2})?$/)
    .nullable(),
  currency: z.string().regex(/^[A-Z]{3}$/),
  acquiredAt: z.string().date().nullable(),
  notes: z.string().max(10000),
  rating: z.number().int().min(0).max(5),
  location: z.string().trim().max(200),
  intent: z.enum(["keep", "trade", "sell"]),
  pressingConfirmed: z.boolean(),
};
export const copySchema = z.object(copyShape).strict();
export const createRecordSchema = z
  .object({
    idempotencyKey: z.string().min(8).max(100),
    candidate: candidateSchema,
    copy: copySchema,
    allowDuplicate: z.boolean(),
    photoIds: z.array(uuid).max(10).optional(),
    crateIds: z.array(uuid).max(100).optional(),
  })
  .strict();
export const updateRecordSchema = copySchema
  .partial()
  .extend({
    version: z.number().int().positive(),
    candidate: candidateSchema.optional(),
    crateIds: z.array(uuid).max(100).optional(),
  })
  .strict();
const wizardSchema = z
  .array(
    z
      .object({
        recordId: uuid,
        version: z.number().int().positive(),
        changes: z
          .object({
            notes: copyShape.notes.optional(),
            rating: copyShape.rating.optional(),
            location: copyShape.location.optional(),
            intent: copyShape.intent.optional(),
            mediaCondition: condition.optional(),
            sleeveCondition: condition.optional(),
          })
          .strict()
          .refine(
            (v) => Object.keys(v).length > 0,
            "An edit must change a field.",
          ),
      })
      .strict(),
  )
  .min(1)
  .max(20);

const dateString = (value: unknown) =>
  value instanceof Date ? value.toISOString() : String(value);
type CopyRow = Record<string, unknown> & {
  id: string;
  release_id: string;
  metadata: CatalogCandidate;
  version: number;
};
const copySelect = `SELECT c.*, COALESCE((SELECT array_agg(ci.crate_id) FROM wax.crate_items ci WHERE ci.copy_id=c.id), '{}') AS crate_ids,
  (SELECT count(*)::int FROM wax.listens l WHERE l.copy_id=c.id AND l.owner_id=c.owner_id) AS spin_count,
  (SELECT max(played_at) FROM wax.listens l WHERE l.copy_id=c.id AND l.owner_id=c.owner_id) AS last_played_at,
  COALESCE((SELECT array_agg(p.id) FROM wax.photos p WHERE p.copy_id=c.id AND p.owner_id=c.owner_id), '{}') AS photo_ids FROM wax.copies c`;
function toCopy(row: CopyRow): RecordCopy {
  return {
    ...row.metadata,
    coverUrl: catalogArtworkPath(row.metadata.discogsId),
    id: row.id,
    releaseId: row.release_id,
    version: row.version,
    mediaCondition: row.media_condition as CopyFields["mediaCondition"],
    sleeveCondition: row.sleeve_condition as CopyFields["sleeveCondition"],
    purchasePrice:
      row.purchase_price == null ? null : String(row.purchase_price),
    currency: String(row.currency),
    acquiredAt: row.acquired_at
      ? dateString(row.acquired_at).slice(0, 10)
      : null,
    notes: String(row.notes),
    rating: Number(row.rating),
    location: String(row.location),
    intent: row.intent as CopyFields["intent"],
    pressingConfirmed: Boolean(row.pressing_confirmed),
    addedAt: dateString(row.created_at),
    updatedAt: dateString(row.updated_at),
    crateIds: row.crate_ids as string[],
    photoIds: row.photo_ids as string[],
    spinCount: Number(row.spin_count),
    lastPlayedAt: row.last_played_at ? dateString(row.last_played_at) : null,
  };
}
export async function getCollector(
  id: string,
  client: Client = getPool(),
): Promise<Collector> {
  const { rows } = await client.query(
    `SELECT u.id,u.name,u.email,EXISTS(SELECT 1 FROM wax.app_owner o WHERE o.user_id=u.id) AS is_owner FROM wax.users u JOIN wax.collectors c ON c.id=u.id WHERE u.id=$1`,
    [id],
  );
  if (!rows[0])
    throw new HttpError(403, "An invitation is required to use WAX.");
  return {
    id: rows[0].id,
    name: rows[0].name,
    email: rows[0].email,
    isOwner: rows[0].is_owner,
  };
}
export async function getCopy(
  userId: string,
  id: string,
  client: Client = getPool(),
): Promise<RecordCopy> {
  uuid.parse(id);
  const { rows } = await client.query<CopyRow>(
    `${copySelect} WHERE c.owner_id=$1 AND c.id=$2`,
    [userId, id],
  );
  if (!rows[0]) throw new HttpError(404, "Record not found.");
  return toCopy(rows[0]);
}
export async function getLibrary(userId: string): Promise<LibraryData> {
  const db = getPool();
  const collector = await getCollector(userId);
  const [copies, crateRows, sharedRows, listenRows] = await Promise.all([
    db.query<CopyRow>(
      `${copySelect} WHERE c.owner_id=$1 ORDER BY c.created_at DESC,c.id`,
      [userId],
    ),
    db.query(
      `SELECT c.*,COALESCE((SELECT jsonb_agg(ci.copy_id) FROM wax.crate_items ci WHERE ci.crate_id=c.id),'[]') AS record_ids,
      COALESCE((SELECT jsonb_agg(jsonb_build_object('userId',u.id,'name',u.name,'email',u.email)) FROM wax.crate_shares s JOIN wax.users u ON u.id=s.recipient_id WHERE s.crate_id=c.id),'[]') AS shares FROM wax.crates c WHERE c.owner_id=$1 ORDER BY c.created_at,c.id`,
      [userId],
    ),
    db.query(
      `SELECT c.id,c.name,c.description,c.color,u.name AS owner_name,
      COALESCE((SELECT jsonb_agg(jsonb_build_object('id',p.id,'metadata',p.metadata,'mediaCondition',p.media_condition,'sleeveCondition',p.sleeve_condition,'pressingConfirmed',p.pressing_confirmed)) FROM wax.crate_items ci JOIN wax.copies p ON p.id=ci.copy_id AND p.owner_id=c.owner_id WHERE ci.crate_id=c.id),'[]') AS records
      FROM wax.crate_shares s JOIN wax.crates c ON c.id=s.crate_id JOIN wax.users u ON u.id=c.owner_id WHERE s.recipient_id=$1 ORDER BY c.name`,
      [userId],
    ),
    db.query(
      `SELECT l.id,l.copy_id,l.played_at,l.notes,c.metadata FROM wax.listens l JOIN wax.copies c ON c.id=l.copy_id AND c.owner_id=l.owner_id WHERE l.owner_id=$1 ORDER BY l.played_at DESC,l.id LIMIT 500`,
      [userId],
    ),
  ]);
  return {
    collector,
    records: copies.rows.map(toCopy),
    crates: crateRows.rows.map((r) => ({
      id: r.id,
      name: r.name,
      description: r.description,
      color: r.color,
      recordIds: r.record_ids,
      shares: r.shares,
    })),
    sharedCrates: sharedRows.rows.map((r) => ({
      id: r.id,
      name: r.name,
      description: r.description,
      color: r.color,
      ownerName: r.owner_name,
      records: r.records.map(
        (p: {
          metadata: CatalogCandidate;
          id: string;
          mediaCondition: SharedRecord["mediaCondition"];
          sleeveCondition: SharedRecord["sleeveCondition"];
          pressingConfirmed: boolean;
        }) => ({
          ...p.metadata,
          coverUrl: catalogArtworkPath(p.metadata.discogsId),
          id: p.id,
          mediaCondition: p.mediaCondition,
          sleeveCondition: p.sleeveCondition,
          pressingConfirmed: p.pressingConfirmed,
        }),
      ),
    })),
    listens: listenRows.rows.map((r) => ({
      id: r.id,
      recordId: r.copy_id,
      artist: r.metadata.artist,
      title: r.metadata.title,
      coverUrl: catalogArtworkPath(r.metadata.discogsId),
      playedAt: dateString(r.played_at),
      notes: r.notes,
    })),
  };
}
async function resolveCandidate(
  candidate: CatalogCandidate,
): Promise<{ metadata: CatalogCandidate; canonical: CatalogCandidate }> {
  if (candidate.discogsId === null) {
    const clean = { ...candidate, coverUrl: null, sourceUrl: null };
    return { metadata: clean, canonical: clean };
  }
  const canonical = candidateSchema.parse(
    await fetchRelease(candidate.discogsId),
  );
  // Copy-level text is the collector's reviewed transcription. Artwork is always read
  // separately from the short-lived provider cache, never from this saved snapshot.
  return {
    canonical,
    metadata: {
      ...candidate,
      discogsId: canonical.discogsId,
      coverUrl: null,
      sourceUrl: canonical.sourceUrl,
    },
  };
}
async function ensureRelease(
  client: PoolClient,
  canonical: CatalogCandidate,
): Promise<string> {
  const { rows } = await client.query(
    `INSERT INTO wax.releases(discogs_id,metadata) VALUES($1,'{}'::jsonb) ON CONFLICT(discogs_id) DO UPDATE SET metadata='{}'::jsonb,fetched_at=now() RETURNING id`,
    [canonical.discogsId],
  );
  return rows[0].id;
}
async function setCopyCrates(
  client: PoolClient,
  userId: string,
  copyId: string,
  crateIds: string[],
) {
  const unique = [...new Set(crateIds)];
  if (unique.length) {
    const found = await client.query(
      `SELECT id FROM wax.crates WHERE owner_id=$1 AND id=ANY($2::uuid[]) FOR UPDATE`,
      [userId, unique],
    );
    if (found.rows.length !== unique.length)
      throw new HttpError(404, "A selected crate was not found.");
  }
  await client.query(
    "DELETE FROM wax.crate_items WHERE owner_id=$1 AND copy_id=$2",
    [userId, copyId],
  );
  for (const crateId of unique)
    await client.query(
      "INSERT INTO wax.crate_items(crate_id,copy_id,owner_id) VALUES($1,$2,$3)",
      [crateId, copyId, userId],
    );
}
export async function createCopy(
  userId: string,
  raw: CreateRecordInput,
): Promise<RecordCopy> {
  const input = createRecordSchema.parse(raw);
  const requestHash = createHash("sha256")
    .update(JSON.stringify(input))
    .digest("hex");
  // Avoid repeated provider requests for completed, identical retries.
  const prior = await getPool().query(
    "SELECT request_hash,result_id FROM wax.mutation_requests WHERE owner_id=$1 AND key=$2",
    [userId, input.idempotencyKey],
  );
  if (prior.rows[0]) {
    if (prior.rows[0].request_hash !== requestHash)
      throw new HttpError(
        409,
        "This save request was already used for different information.",
      );
    if (prior.rows[0].result_id)
      return getCopy(userId, prior.rows[0].result_id);
  }
  const { metadata, canonical } = await resolveCandidate(input.candidate);
  return withTransaction(async (client) => {
    await lockCollection(client, userId);
    await client.query(
      `INSERT INTO wax.mutation_requests(owner_id,key,request_hash) VALUES($1,$2,$3) ON CONFLICT DO NOTHING`,
      [userId, input.idempotencyKey, requestHash],
    );
    const claimed = (
      await client.query(
        "SELECT request_hash,result_id FROM wax.mutation_requests WHERE owner_id=$1 AND key=$2 FOR UPDATE",
        [userId, input.idempotencyKey],
      )
    ).rows[0];
    if (claimed.request_hash !== requestHash)
      throw new HttpError(
        409,
        "This save request was already used for different information.",
      );
    if (claimed.result_id) return getCopy(userId, claimed.result_id, client);
    const duplicates = await client.query(
      `SELECT id FROM wax.copies WHERE owner_id=$1 AND (($2::bigint IS NOT NULL AND metadata->>'discogsId'=$2::text) OR ($2::bigint IS NULL AND lower(metadata->>'artist')=lower($3) AND lower(metadata->>'title')=lower($4)))`,
      [userId, metadata.discogsId, metadata.artist, metadata.title],
    );
    if (!input.allowDuplicate && duplicates.rows.length)
      throw new HttpError(
        409,
        "You already have this record. Add another physical copy?",
        { duplicateIds: duplicates.rows.map((r) => r.id) },
      );
    const releaseId = await ensureRelease(client, canonical);
    const f = input.copy;
    const row = (
      await client.query(
        `INSERT INTO wax.copies(owner_id,release_id,metadata,media_condition,sleeve_condition,purchase_price,currency,acquired_at,notes,rating,location,intent,pressing_confirmed) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING id`,
        [
          userId,
          releaseId,
          JSON.stringify(metadata),
          f.mediaCondition,
          f.sleeveCondition,
          f.purchasePrice,
          f.currency,
          f.acquiredAt,
          f.notes,
          f.rating,
          f.location,
          f.intent,
          metadata.discogsId !== null && f.pressingConfirmed,
        ],
      )
    ).rows[0];
    await setCopyCrates(client, userId, row.id, input.crateIds ?? []);
    const photos = [...new Set(input.photoIds ?? [])];
    if (photos.length) {
      const linked = await client.query(
        `UPDATE wax.photos SET copy_id=$1 WHERE owner_id=$2 AND id=ANY($3::uuid[]) AND copy_id IS NULL RETURNING id`,
        [row.id, userId, photos],
      );
      if (linked.rows.length !== photos.length)
        throw new HttpError(
          409,
          "A photo is unavailable or already belongs to a record.",
        );
    }
    await client.query(
      "UPDATE wax.mutation_requests SET result_id=$3 WHERE owner_id=$1 AND key=$2",
      [userId, input.idempotencyKey, row.id],
    );
    return getCopy(userId, row.id, client);
  });
}
const fieldColumns: Record<keyof CopyFields, string> = {
  mediaCondition: "media_condition",
  sleeveCondition: "sleeve_condition",
  purchasePrice: "purchase_price",
  currency: "currency",
  acquiredAt: "acquired_at",
  notes: "notes",
  rating: "rating",
  location: "location",
  intent: "intent",
  pressingConfirmed: "pressing_confirmed",
};
async function applyUpdate(
  client: PoolClient,
  userId: string,
  id: string,
  input: UpdateRecordInput,
  resolved?: Awaited<ReturnType<typeof resolveCandidate>>,
) {
  const existing = (
    await client.query<CopyRow>(
      "SELECT * FROM wax.copies WHERE owner_id=$1 AND id=$2 FOR UPDATE",
      [userId, id],
    )
  ).rows[0];
  if (!existing) throw new HttpError(404, "Record not found.");
  if (existing.version !== input.version)
    throw new HttpError(409, "This record changed. Refresh it before saving.");
  const values: unknown[] = [userId, id];
  const sets: string[] = [];
  const selectedMetadata = resolved?.metadata ?? existing.metadata;
  const fields = { ...input };
  if (selectedMetadata.discogsId === null) fields.pressingConfirmed = false;
  for (const [key, column] of Object.entries(fieldColumns)) {
    if (key in fields) {
      values.push(fields[key as keyof CopyFields]);
      sets.push(`${column}=$${values.length}`);
    }
  }
  if (resolved) {
    const releaseId = await ensureRelease(client, resolved.canonical);
    values.push(releaseId, JSON.stringify(resolved.metadata));
    sets.push(`release_id=$${values.length - 1}`, `metadata=$${values.length}`);
  }
  sets.push("version=version+1", "updated_at=now()");
  await client.query(
    `UPDATE wax.copies SET ${sets.join(",")} WHERE owner_id=$1 AND id=$2`,
    values,
  );
  if (input.crateIds) await setCopyCrates(client, userId, id, input.crateIds);
}
export async function updateCopy(
  userId: string,
  id: string,
  raw: UpdateRecordInput,
): Promise<RecordCopy> {
  uuid.parse(id);
  const input = updateRecordSchema.parse(raw);
  // Authorize before any provider lookup.
  await getCopy(userId, id);
  const resolved = input.candidate
    ? await resolveCandidate(input.candidate)
    : undefined;
  return withTransaction(async (client) => {
    await lockCollection(client, userId);
    await applyUpdate(client, userId, id, input, resolved);
    return getCopy(userId, id, client);
  });
}
export async function approveChanges(
  userId: string,
  raw: WizardChange[],
  client?: PoolClient,
): Promise<void> {
  const changes = wizardSchema.parse(raw);
  if (new Set(changes.map((c) => c.recordId)).size !== changes.length)
    throw new HttpError(400, "A proposal cannot edit a record twice.");
  const apply = async (tx: PoolClient) => {
    await lockCollection(tx, userId);
    for (const change of [...changes].sort((a, b) =>
      a.recordId.localeCompare(b.recordId),
    ))
      await applyUpdate(tx, userId, change.recordId, {
        ...change.changes,
        version: change.version,
      });
  };
  if (client) await apply(client);
  else await withTransaction(apply);
}
export async function deleteCopy(userId: string, id: string, version: number) {
  uuid.parse(id);
  z.number().int().positive().parse(version);
  return withTransaction(async (client) => {
    await lockCollection(client, userId);
    const copy = (
      await client.query(
        "SELECT version FROM wax.copies WHERE owner_id=$1 AND id=$2 FOR UPDATE",
        [userId, id],
      )
    ).rows[0];
    if (!copy) throw new HttpError(404, "Record not found.");
    if (copy.version !== version)
      throw new HttpError(409, "This record changed. Refresh before deleting.");
    await client.query("DELETE FROM wax.copies WHERE owner_id=$1 AND id=$2", [
      userId,
      id,
    ]);
    return { ok: true };
  });
}
const crateSchema = z
  .object({
    name: z.string().trim().min(1).max(100),
    description: z.string().max(2000).default(""),
    color: z
      .string()
      .regex(/^#[0-9a-fA-F]{6}$/)
      .default("#d85a2a"),
  })
  .strict();
export async function createCrate(
  userId: string,
  raw: unknown,
): Promise<Crate> {
  const input = crateSchema.parse(raw);
  const r = (
    await getPool().query(
      `INSERT INTO wax.crates(owner_id,name,description,color) VALUES($1,$2,$3,$4) RETURNING id`,
      [userId, input.name, input.description, input.color],
    )
  ).rows[0];
  return { id: r.id, ...input, recordIds: [], shares: [] };
}
export async function updateCrate(
  userId: string,
  id: string,
  raw: unknown,
): Promise<Crate> {
  uuid.parse(id);
  const input = crateSchema
    .partial()
    .extend({ recordIds: z.array(uuid).max(10000).optional() })
    .strict()
    .parse(raw);
  await withTransaction(async (client) => {
    // Copy edits and crate edits take the same owner lock before their row locks.
    // This prevents opposite copy/crate lock ordering from deadlocking concurrent saves.
    await lockCollection(client, userId);
    const crate = (
      await client.query(
        "SELECT id FROM wax.crates WHERE id=$1 AND owner_id=$2 FOR UPDATE",
        [id, userId],
      )
    ).rows[0];
    if (!crate) throw new HttpError(404, "Crate not found.");
    for (const key of ["name", "description", "color"] as const)
      if (input[key] !== undefined)
        await client.query(
          `UPDATE wax.crates SET ${key}=$3 WHERE id=$1 AND owner_id=$2`,
          [id, userId, input[key]],
        );
    if (input.recordIds) {
      const ids = [...new Set(input.recordIds)];
      const found = await client.query(
        "SELECT id FROM wax.copies WHERE owner_id=$1 AND id=ANY($2::uuid[]) FOR UPDATE",
        [userId, ids],
      );
      if (found.rows.length !== ids.length)
        throw new HttpError(404, "A selected record was not found.");
      await client.query(
        "DELETE FROM wax.crate_items WHERE crate_id=$1 AND owner_id=$2",
        [id, userId],
      );
      for (const copyId of ids)
        await client.query(
          "INSERT INTO wax.crate_items(crate_id,copy_id,owner_id) VALUES($1,$2,$3)",
          [id, copyId, userId],
        );
    }
  });
  const updated = (await getLibrary(userId)).crates.find((c) => c.id === id);
  if (!updated) throw new HttpError(404, "Crate not found.");
  return updated;
}
export async function deleteCrate(userId: string, id: string) {
  uuid.parse(id);
  return withTransaction(async (client) => {
    await lockCollection(client, userId);
    const r = await client.query(
      "DELETE FROM wax.crates WHERE owner_id=$1 AND id=$2 RETURNING id",
      [userId, id],
    );
    if (!r.rows.length) throw new HttpError(404, "Crate not found.");
    return { ok: true };
  });
}
export async function shareCrate(userId: string, id: string, email: string) {
  uuid.parse(id);
  const normalized = z.string().trim().email().parse(email).toLowerCase();
  return withTransaction(async (client) => {
    const crate = (
      await client.query(
        "SELECT id FROM wax.crates WHERE owner_id=$1 AND id=$2 FOR UPDATE",
        [userId, id],
      )
    ).rows[0];
    if (!crate) throw new HttpError(404, "Crate not found.");
    const recipient = (
      await client.query(
        "SELECT u.id,u.name,u.email FROM wax.users u JOIN wax.collectors c ON c.id=u.id WHERE lower(u.email)=$1",
        [normalized],
      )
    ).rows[0];
    if (!recipient || recipient.id === userId)
      throw new HttpError(
        400,
        "Choose another collector who already has a WAX account.",
      );
    await client.query(
      "INSERT INTO wax.crate_shares(crate_id,recipient_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
      [id, recipient.id],
    );
    return {
      userId: recipient.id,
      name: recipient.name,
      email: recipient.email,
    };
  });
}
export async function unshareCrate(
  userId: string,
  id: string,
  recipientId: string,
) {
  uuid.parse(id);
  z.string().min(1).max(128).parse(recipientId);
  const r = await getPool().query(
    "DELETE FROM wax.crate_shares s USING wax.crates c WHERE s.crate_id=c.id AND c.owner_id=$1 AND c.id=$2 AND s.recipient_id=$3 RETURNING s.crate_id",
    [userId, id, recipientId],
  );
  if (!r.rows.length) throw new HttpError(404, "Share not found.");
  return { ok: true };
}
export async function logListen(userId: string, raw: unknown) {
  const input = z
    .object({
      recordId: uuid,
      notes: z.string().max(10000).default(""),
      playedAt: z.string().datetime({ offset: true }).optional(),
    })
    .strict()
    .parse(raw);
  const copy = await getCopy(userId, input.recordId);
  const r = (
    await getPool().query(
      `INSERT INTO wax.listens(owner_id,copy_id,notes,played_at) VALUES($1,$2,$3,COALESCE($4::timestamptz,now())) RETURNING id,played_at`,
      [userId, input.recordId, input.notes, input.playedAt ?? null],
    )
  ).rows[0];
  return {
    id: r.id,
    recordId: input.recordId,
    artist: copy.artist,
    title: copy.title,
    coverUrl: copy.coverUrl,
    notes: input.notes,
    playedAt: dateString(r.played_at),
  };
}
export function csvCell(value: unknown): string {
  const text = String(value ?? "");
  const safe =
    /^[\s]*[=+@-]/.test(text) || /^[\t\r\n]/.test(text) ? `'${text}` : text;
  return `"${safe.replaceAll('"', '""')}"`;
}
export async function exportCollection(userId: string): Promise<Response> {
  const { records } = await getLibrary(userId);
  const fields = [
    "id",
    "artist",
    "title",
    "year",
    "label",
    "catalogNumber",
    "country",
    "format",
    "mediaCondition",
    "sleeveCondition",
    "purchasePrice",
    "currency",
    "acquiredAt",
    "rating",
    "location",
    "intent",
    "notes",
  ] as const;
  const csv = [
    fields.map(csvCell).join(","),
    ...records.map((r) => fields.map((f) => csvCell(r[f])).join(",")),
  ].join("\r\n");
  return new Response("\uFEFF" + csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="wax-collection.csv"',
      "Cache-Control": "private, no-store",
    },
  });
}
