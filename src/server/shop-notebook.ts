import { z } from "zod";
import {
  SHOP_AVATARS,
  type SavedShop,
  type SaveShopInput,
  type ShopEmployee,
  type ShopNotebook,
  type ShopSnapshot,
} from "@/lib/shop-notebook";
import { getPool } from "./db";
import { HttpError } from "./http";

const boundedText = (maximum: number) =>
  z
    .string()
    .trim()
    .min(1)
    .max(maximum)
    .regex(/^[^\x00-\x1f\x7f]+$/);

export const notebookShopIdSchema = z
  .string()
  .max(90)
  .regex(
    /^(?:(?:node|way|relation)\/[1-9]\d*|directory\/[a-z0-9]+(?:-[a-z0-9]+)*)$/,
  )
  .refine(
    (value) =>
      value.startsWith("directory/") ||
      Number.isSafeInteger(Number(value.split("/")[1])),
  );

const websiteSchema = z
  .string()
  .trim()
  .max(2048)
  .refine((value) => {
    try {
      const url = new URL(value);
      return (
        ["https:", "http:"].includes(url.protocol) &&
        !url.username &&
        !url.password &&
        !/[\x00-\x20\x7f]/.test(value)
      );
    } catch {
      return false;
    }
  });

const shopSchema = z
  .object({
    id: notebookShopIdSchema,
    name: boundedText(240),
    latitude: z.number().finite().min(-90).max(90),
    longitude: z.number().finite().min(-180).max(180),
    address: boundedText(500).nullable(),
    website: websiteSchema.nullable(),
  })
  .strict();

const employeeSchema = z
  .object({
    id: z.uuid(),
    name: boundedText(80),
    role: boundedText(80),
    avatar: z.enum(SHOP_AVATARS.map((avatar) => avatar.id)),
  })
  .strict();

const versionSchema = z.number().int().positive().max(2_147_483_646);

export const saveShopSchema = z
  .object({
    shop: shopSchema,
    rating: z.number().int().min(1).max(5).nullable(),
    employees: z
      .array(employeeSchema)
      .max(30)
      .refine(
        (employees) =>
          new Set(employees.map((employee) => employee.id.toLowerCase()))
            .size === employees.length,
        "Each employee must have a different ID.",
      ),
    version: versionSchema.nullable(),
  })
  .strict();

export const deleteShopSchema = z
  .object({ shopId: notebookShopIdSchema, version: versionSchema })
  .strict();

export async function readNotebookRequest(request: Request): Promise<unknown> {
  const maximum = 32 * 1024;
  const tooLarge = () =>
    new HttpError(413, "These shop notes are too large. Please shorten them.");
  if (Number(request.headers.get("content-length")) > maximum) throw tooLarge();
  const reader = request.body?.getReader();
  if (!reader) throw new HttpError(400, "The shop notes could not be read.");
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximum) {
        await reader.cancel();
        throw tooLarge();
      }
      chunks.push(value);
    }
    return JSON.parse(Buffer.concat(chunks, length).toString("utf8"));
  } finally {
    reader.releaseLock();
  }
}

type NotebookRow = {
  shop: ShopSnapshot;
  rating: number | null;
  employees: ShopEmployee[];
  version: number;
  updated_at: Date | string;
};

const toSavedShop = (row: NotebookRow): SavedShop => ({
  shop: row.shop,
  rating: row.rating,
  employees: row.employees,
  version: row.version,
  updatedAt: new Date(row.updated_at).toISOString(),
});

const conflict = () =>
  new HttpError(
    409,
    "This shop was changed in another tab. Reload its saved details before saving again. Your draft is still here.",
  );

export async function getShopNotebook(ownerId: string): Promise<ShopNotebook> {
  const result = await getPool().query<NotebookRow>(
    `SELECT shop,rating,employees,version,updated_at FROM wax.shop_notebook
     WHERE owner_id=$1 ORDER BY updated_at DESC,shop_id`,
    [ownerId],
  );
  return { shops: result.rows.map(toSavedShop) };
}

export async function saveShopNotebook(
  ownerId: string,
  input: SaveShopInput,
): Promise<SavedShop> {
  const values = [
    ownerId,
    input.shop.id,
    JSON.stringify(input.shop),
    input.rating,
    JSON.stringify(input.employees),
  ];
  if (input.version === null) {
    const result = await getPool().query<NotebookRow>(
      `INSERT INTO wax.shop_notebook(owner_id,shop_id,shop,rating,employees)
       VALUES($1,$2,$3::jsonb,$4,$5::jsonb)
       ON CONFLICT(owner_id,shop_id) DO NOTHING
       RETURNING shop,rating,employees,version,updated_at`,
      values,
    );
    if (!result.rows[0]) throw conflict();
    return toSavedShop(result.rows[0]);
  }
  const result = await getPool().query<NotebookRow>(
    `UPDATE wax.shop_notebook SET shop=$3::jsonb,rating=$4,employees=$5::jsonb,
       version=version+1,updated_at=clock_timestamp()
     WHERE owner_id=$1 AND shop_id=$2 AND version=$6
     RETURNING shop,rating,employees,version,updated_at`,
    [...values, input.version],
  );
  if (!result.rows[0]) await throwMissingOrConflict(ownerId, input.shop.id);
  return toSavedShop(result.rows[0]);
}

async function throwMissingOrConflict(
  ownerId: string,
  shopId: string,
): Promise<never> {
  const exists = await getPool().query(
    "SELECT 1 FROM wax.shop_notebook WHERE owner_id=$1 AND shop_id=$2",
    [ownerId, shopId],
  );
  if (!exists.rows.length)
    throw new HttpError(404, "This shop is not in your saved shops.");
  throw conflict();
}

export async function deleteShopNotebook(
  ownerId: string,
  input: z.infer<typeof deleteShopSchema>,
): Promise<{ ok: true }> {
  const result = await getPool().query(
    `DELETE FROM wax.shop_notebook WHERE owner_id=$1 AND shop_id=$2 AND version=$3
     RETURNING shop_id`,
    [ownerId, input.shopId, input.version],
  );
  if (!result.rows.length) await throwMissingOrConflict(ownerId, input.shopId);
  return { ok: true };
}
