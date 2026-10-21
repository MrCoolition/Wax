import { readFile } from "node:fs/promises";
import { PGlite } from "@electric-sql/pglite";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { EMPTY_CANDIDATE } from "../src/lib/contracts";

const state = vi.hoisted(() => ({
  db: null as PGlite | null,
  generate: vi.fn(),
  blobGet: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
  withTransaction: <T>(
    action: (client: {
      query: (sql: string, params?: unknown[]) => Promise<unknown>;
    }) => Promise<T>,
  ) =>
    state.db!.transaction(async (tx) =>
      action({ query: (sql, params) => tx.query(sql, params) }),
    ),
}));
vi.mock("ai", async (original) => ({
  ...(await original<typeof import("ai")>()),
  generateText: state.generate,
}));
vi.mock("@vercel/blob", () => ({
  get: state.blobGet,
  put: vi.fn(),
  del: vi.fn(),
}));
import { askWizard, decideProposal } from "../src/server/wizard";
import { photoResponse } from "../src/server/photos";

const first = "10000000-0000-4000-8000-000000000001";
const second = "20000000-0000-4000-8000-000000000002";
const other = "30000000-0000-4000-8000-000000000003";
const photo = "40000000-0000-4000-8000-000000000004";
const changes = {
  notes: null,
  rating: null,
  location: "Shelf B",
  intent: null,
  mediaCondition: null,
  sleeveCondition: null,
};
async function copy(id: string) {
  return (
    await state.db!.query<{ location: string; version: number }>(
      "SELECT location,version FROM wax.copies WHERE id=$1",
      [id],
    )
  ).rows[0];
}
async function seedProposal(edits: unknown[]) {
  return (
    await state.db!.query<{ id: string }>(
      "INSERT INTO wax.wizard_proposals(owner_id,summary,changes) VALUES('alice','Move records',$1::jsonb) RETURNING id",
      [JSON.stringify(edits)],
    )
  ).rows[0].id;
}

describe("integration permissions and reviewed edits", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    await state.db.exec(
      await readFile(
        new URL("../migrations/0001_core.sql", import.meta.url),
        "utf8",
      ),
    );
    await state.db.exec(
      await readFile(
        new URL("../migrations/0002_integrations.sql", import.meta.url),
        "utf8",
      ),
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.stubEnv("OPENAI_API_KEY", "fixture-key-never-used");
    await state.db!.exec("TRUNCATE wax.users,wax.releases CASCADE");
    await state.db!.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('alice','Alice','alice@example.test'),('bob','Bob','bob@example.test'); INSERT INTO wax.collectors(id) VALUES('alice'),('bob');",
    );
    for (const [id, owner, title] of [
      [first, "alice", "First album"],
      [second, "alice", "Second album"],
      [other, "bob", "BOB_PRIVATE_ALBUM"],
    ]) {
      const metadata = JSON.stringify({
        ...EMPTY_CANDIDATE,
        artist: "Fixture artist",
        title,
      });
      const release = (
        await state.db!.query<{ id: string }>(
          "INSERT INTO wax.releases(metadata) VALUES($1::jsonb) RETURNING id",
          [metadata],
        )
      ).rows[0].id;
      await state.db!.query(
        "INSERT INTO wax.copies(id,owner_id,release_id,metadata,location) VALUES($1,$2,$3,$4::jsonb,'Shelf A')",
        [id, owner, release, metadata],
      );
    }
  });
  afterAll(async () => {
    vi.unstubAllEnvs();
    await state.db?.close();
  });

  it("creates a pending proposal without edits, then applies approval exactly once", async () => {
    state.generate.mockResolvedValue({
      output: {
        reply: "Review this shelf move.",
        recordIds: [first],
        proposal: {
          summary: "Move first album",
          changes: [{ recordId: first, version: 1, changes }],
        },
      },
    });
    const reply = await askWizard("alice", {
      message: "Move my first album to Shelf B.",
    });
    expect(await copy(first)).toEqual({ location: "Shelf A", version: 1 });
    expect(reply.proposal?.id).toBeTruthy();
    await expect(
      decideProposal("bob", reply.proposal!.id, "approve"),
    ).rejects.toMatchObject({ status: 404 });
    await decideProposal("alice", reply.proposal!.id, "approve");
    await decideProposal("alice", reply.proposal!.id, "approve");
    expect(await copy(first)).toEqual({ location: "Shelf B", version: 2 });
    const prompt = JSON.stringify(state.generate.mock.calls[0][0].messages);
    expect(prompt).not.toContain("BOB_PRIVATE_ALBUM");
    expect(prompt).not.toContain(other);
  });

  it("rolls back all copies when one reviewed version is stale", async () => {
    const proposal = await seedProposal([
      { recordId: first, version: 1, changes: { location: "Shelf B" } },
      { recordId: second, version: 1, changes: { location: "Shelf B" } },
    ]);
    await state.db!.query("UPDATE wax.copies SET version=2 WHERE id=$1", [
      second,
    ]);
    await expect(
      decideProposal("alice", proposal, "approve"),
    ).rejects.toMatchObject({ status: 409 });
    expect(await copy(first)).toEqual({ location: "Shelf A", version: 1 });
    const status = (
      await state.db!.query<{ status: string }>(
        "SELECT status FROM wax.wizard_proposals WHERE id=$1",
        [proposal],
      )
    ).rows[0].status;
    expect(status).toBe("pending");
  });

  it("rejects a proposal without changing any copy, including repeated rejection", async () => {
    const proposal = await seedProposal([
      { recordId: first, version: 1, changes: { rating: 5 } },
    ]);
    await decideProposal("alice", proposal, "reject");
    await decideProposal("alice", proposal, "reject");
    await expect(
      decideProposal("alice", proposal, "approve"),
    ).rejects.toMatchObject({ status: 409 });
    expect(await copy(first)).toEqual({ location: "Shelf A", version: 1 });
  });

  it("denies another collector’s photo before accessing private storage", async () => {
    await state.db!.query(
      "INSERT INTO wax.photos(id,owner_id,copy_id,pathname,content_type,byte_size) VALUES($1,'alice',$2,'wax/alice/private.jpg','image/jpeg',100)",
      [photo, first],
    );
    await expect(photoResponse("bob", photo)).rejects.toMatchObject({
      status: 404,
    });
    expect(state.blobGet).not.toHaveBeenCalled();
  });
});
