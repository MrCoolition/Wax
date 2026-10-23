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
  collector: vi.fn(),
  year: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
}));
vi.mock("../src/server/auth", () => ({ requireCollector: state.collector }));
vi.mock("../src/server/catalog", async (original) => ({
  ...(await original<typeof import("../src/server/catalog")>()),
  fetchOriginalYear: state.year,
}));
import { POST } from "../src/app/api/records/years/route";
import { getLibrary, exportCollection } from "../src/server/collection";
import { HttpError } from "../src/server/http";

const request = (body: unknown = {}, origin = "https://wax.example.test") =>
  new Request("https://wax.example.test/api/records/years", {
    method: "POST",
    headers: { Origin: origin, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
async function legacy(owner = "alice", discogsId: number | null = 42) {
  const metadata = {
    ...EMPTY_CANDIDATE,
    artist: "The Beach Boys",
    title: "Pet Sounds",
    discogsId,
    year: 2024,
  };
  delete metadata.originalYear;
  const release = (
    await state.db!.query<{ id: string }>(
      "INSERT INTO wax.releases(metadata) VALUES('{}') RETURNING id",
    )
  ).rows[0];
  return (
    await state.db!.query<{ id: string }>(
      `INSERT INTO wax.copies(owner_id,release_id,metadata,notes,purchase_price,location,rating)
      VALUES($1,$2,$3::jsonb,'Private notes',25.50,'Top shelf',5) RETURNING id`,
      [owner, release.id, JSON.stringify(metadata)],
    )
  ).rows[0].id;
}
async function row(id: string) {
  return (
    await state.db!.query<{
      metadata: { year: number; originalYear?: number | null };
      version: number;
      notes: string;
      purchase_price: string;
      location: string;
      rating: number;
    }>("SELECT * FROM wax.copies WHERE id=$1", [id])
  ).rows[0];
}

describe("bounded original album year enrichment", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    for (const file of ["0001_core.sql", "0002_integrations.sql"])
      await state.db.exec(
        await readFile(
          new URL(`../migrations/${file}`, import.meta.url),
          "utf8",
        ),
      );
    await state.db.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('alice','Alice','alice@example.test'),('bob','Bob','bob@example.test'); INSERT INTO wax.collectors(id) VALUES('alice'),('bob')",
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    state.collector.mockResolvedValue({ id: "alice" });
    state.year.mockResolvedValue({ originalYear: 1966, resolved: true });
    await state.db!.exec(
      "TRUNCATE wax.copies,wax.releases,wax.integration_limits CASCADE",
    );
  });
  afterAll(async () => {
    await state.db?.close();
  });

  it("requires sign-in and same-origin requests, and rejects caller-selected owners or copies", async () => {
    const id = await legacy();
    expect((await POST(request({}, "https://evil.test"))).status).toBe(403);
    expect(state.collector).not.toHaveBeenCalled();
    state.collector.mockRejectedValueOnce(
      new HttpError(401, "Please sign in."),
    );
    expect((await POST(request())).status).toBe(401);
    expect((await POST(request({ ownerId: "bob" }))).status).toBe(400);
    expect((await POST(request({ ids: [id] }))).status).toBe(400);
    expect((await POST(request({ extra: "x".repeat(1024) }))).status).toBe(400);
    expect(state.year).not.toHaveBeenCalled();
  });

  it("corrects only the owner's missing album year and preserves pressing and personal data", async () => {
    const aliceId = await legacy();
    const bobId = await legacy("bob", 43);
    const before = await row(aliceId);
    const response = await POST(request());
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.json()).toEqual({
      updated: 1,
      processed: 1,
      remaining: 0,
      transientFailure: false,
    });
    const after = await row(aliceId);
    expect(after.metadata).toEqual({ ...before.metadata, originalYear: 1966 });
    expect(after).toMatchObject({
      version: 2,
      notes: before.notes,
      purchase_price: before.purchase_price,
      location: before.location,
      rating: before.rating,
    });
    expect((await row(bobId)).metadata.originalYear).toBeUndefined();
    expect(state.year).toHaveBeenCalledExactlyOnceWith(42);
    expect(await (await POST(request())).json()).toMatchObject({
      updated: 0,
      processed: 0,
      remaining: 0,
    });
  });

  it("library reads signal missing owner data without a provider waterfall, and CSV labels both years", async () => {
    const id = await legacy();
    expect(await getLibrary("alice")).toMatchObject({
      needsOriginalYears: true,
      records: [{ id, year: 2024, originalYear: null }],
    });
    expect(state.year).not.toHaveBeenCalled();
    await POST(request());
    expect(await getLibrary("alice")).toMatchObject({
      needsOriginalYears: false,
      records: [{ id, year: 2024, originalYear: 1966 }],
    });
    const csv = await (await exportCollection("alice")).text();
    expect(csv).toContain('"originalYear","pressingYear"');
    expect(csv).toContain('"1966","2024"');
  });

  it("records genuine unknowns once but leaves transient failures eligible for retry", async () => {
    const id = await legacy();
    state.year.mockResolvedValueOnce({ originalYear: null, resolved: false });
    expect(await (await POST(request())).json()).toEqual({
      updated: 0,
      processed: 0,
      remaining: 1,
      transientFailure: true,
    });
    expect((await row(id)).metadata.originalYear).toBeUndefined();
    state.year.mockResolvedValueOnce({ originalYear: null, resolved: true });
    expect(await (await POST(request())).json()).toMatchObject({
      updated: 1,
      remaining: 0,
      transientFailure: false,
    });
    expect((await row(id)).metadata).toMatchObject({
      year: 2024,
      originalYear: null,
    });
    const manual = await legacy("alice", null);
    expect(await (await POST(request())).json()).toMatchObject({
      updated: 1,
      remaining: 0,
    });
    expect((await row(manual)).metadata.originalYear).toBeNull();
    expect(state.year).toHaveBeenCalledTimes(2);
  });

  it("does not overwrite a concurrent edit or already reviewed year", async () => {
    const id = await legacy();
    state.year.mockImplementationOnce(async () => {
      await state.db!.query(
        "UPDATE wax.copies SET metadata=jsonb_set(metadata,'{originalYear}','1967'),notes='Changed while loading',version=version+1 WHERE id=$1",
        [id],
      );
      return { originalYear: 1966, resolved: true };
    });
    expect(await (await POST(request())).json()).toMatchObject({
      updated: 0,
      processed: 1,
      remaining: 0,
    });
    expect(await row(id)).toMatchObject({
      metadata: { originalYear: 1967, year: 2024 },
      notes: "Changed while loading",
      version: 2,
    });
  });

  it("bounds each request to eight records and applies collector limits", async () => {
    for (let index = 0; index < 10; index++) await legacy("alice", index + 1);
    expect(await (await POST(request())).json()).toMatchObject({
      updated: 8,
      processed: 8,
      remaining: 2,
    });
    expect(state.year).toHaveBeenCalledTimes(8);
    await state.db!.query(
      "UPDATE wax.integration_limits SET count=10 WHERE owner_id='alice' AND operation='original-years'",
    );
    expect((await POST(request())).status).toBe(429);
    expect(state.year).toHaveBeenCalledTimes(8);
  });
});
