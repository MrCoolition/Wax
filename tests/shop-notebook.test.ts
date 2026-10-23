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
import type { SaveShopInput, SavedShop } from "../src/lib/shop-notebook";

const state = vi.hoisted(() => ({
  db: null as PGlite | null,
  collector: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
}));
vi.mock("../src/server/auth", () => ({ requireCollector: state.collector }));

import { HttpError } from "../src/server/http";
import {
  saveShopSchema,
  notebookShopIdSchema,
} from "../src/server/shop-notebook";
import { GET, PUT, DELETE } from "../src/app/api/shops/notebook/route";

const initial: SaveShopInput = {
  shop: {
    id: "node/123456",
    name: "  Good Records  ",
    latitude: 39.96,
    longitude: -83,
    address: "10 High Street, Columbus",
    website: "https://records.example.test/",
  },
  rating: 5,
  employees: [
    {
      id: "1e1871a8-08d7-45c2-b095-ef35e4e8faf0",
      name: "  Jordan  ",
      role: "  Jazz buyer  ",
      avatar: "groove",
    },
  ],
  version: null,
};
const request = (
  method: "GET" | "PUT" | "DELETE",
  body?: unknown,
  origin: string | null = "https://wax.example.test",
) =>
  new Request("https://wax.example.test/api/shops/notebook", {
    method,
    headers: {
      ...(origin ? { Origin: origin } : {}),
      "Content-Type": "application/json",
    },
    ...(method === "GET" ? {} : { body: JSON.stringify(body) }),
  });
const read = async (): Promise<{ shops: SavedShop[] }> => {
  const response = await GET(request("GET"));
  expect(response.status).toBe(200);
  expect(response.headers.get("cache-control")).toBe("private, no-store");
  return response.json();
};
const save = async (input: SaveShopInput = initial): Promise<SavedShop> => {
  const response = await PUT(request("PUT", input));
  expect(response.status).toBe(200);
  return response.json();
};

describe("private shop notebook", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    for (const file of [
      "0001_core.sql",
      "0002_integrations.sql",
      "0005_shop_discovery.sql",
      "0006_shop_notebook.sql",
      "0007_shop_directory.sql",
    ])
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
    await state.db!.exec(
      "TRUNCATE wax.shop_notebook,wax.integration_limits,wax.shop_discovery_cache",
    );
  });
  afterAll(async () => {
    await state.db?.close();
  });

  it("requires sign-in for reads and writes", async () => {
    state.collector.mockRejectedValue(new HttpError(401, "Please sign in."));
    for (const response of [
      await GET(request("GET")),
      await PUT(request("PUT", initial)),
      await DELETE(request("DELETE", { shopId: initial.shop.id, version: 1 })),
    ])
      expect(response.status).toBe(401);
    expect(
      (await state.db!.query("SELECT * FROM wax.shop_notebook")).rows,
    ).toEqual([]);
  });

  it("rejects cross-origin and missing-origin mutations before changing data", async () => {
    for (const origin of [null, "https://elsewhere.example.test"]) {
      expect((await PUT(request("PUT", initial, origin))).status).toBe(403);
      expect(
        (
          await DELETE(
            request("DELETE", { shopId: initial.shop.id, version: 1 }, origin),
          )
        ).status,
      ).toBe(403);
    }
    expect(state.collector).not.toHaveBeenCalled();
    expect((await read()).shops).toEqual([]);
  });

  it("persists normalized ratings, staff names, roles, and selected avatars", async () => {
    const saved = await save();
    expect(saved).toMatchObject({
      shop: { id: initial.shop.id, name: "Good Records" },
      rating: 5,
      employees: [{ name: "Jordan", role: "Jazz buyer", avatar: "groove" }],
      version: 1,
    });
    expect(new Date(saved.updatedAt).toISOString()).toBe(saved.updatedAt);
    expect((await read()).shops).toEqual([saved]);
    expect(
      (await state.db!.query("SELECT * FROM wax.shop_discovery_cache")).rows,
    ).toEqual([]);
    expect(saved).not.toHaveProperty("owner_id");
  });

  it("isolates two collectors' ratings and staff lists for the same shop", async () => {
    const alice = await save();
    state.collector.mockResolvedValue({ id: "bob" });
    expect((await read()).shops).toEqual([]);
    expect(
      (await PUT(request("PUT", { ...initial, version: alice.version })))
        .status,
    ).toBe(404);
    expect(
      (
        await DELETE(
          request("DELETE", {
            shopId: initial.shop.id,
            version: alice.version,
          }),
        )
      ).status,
    ).toBe(404);
    const bob = await save({ ...initial, rating: 2, employees: [] });
    expect((await read()).shops).toEqual([bob]);
    state.collector.mockResolvedValue({ id: "alice" });
    expect((await read()).shops).toEqual([alice]);
    expect(
      (await PUT(request("PUT", { ...initial, ownerId: "bob", version: 1 })))
        .status,
    ).toBe(400);
    expect(
      (
        await DELETE(
          request("DELETE", {
            shopId: initial.shop.id,
            version: 1,
            owner_id: "bob",
          }),
        )
      ).status,
    ).toBe(400);
    const disguisedRead = await GET(
      new Request("https://wax.example.test/api/shops/notebook?ownerId=bob"),
    );
    expect((await disguisedRead.json()).shops).toEqual([alice]);
  });

  it("persists private directory-shop notes without changing existing OSM identities", async () => {
    const osm = await save();
    const input = {
      ...initial,
      shop: { ...initial.shop, id: "directory/spoonful-records-columbus" },
    };
    const directory = await save(input);
    expect((await read()).shops).toEqual([directory, osm]);
    state.collector.mockResolvedValue({ id: "bob" });
    expect((await read()).shops).toEqual([]);
    expect((await PUT(request("PUT", { ...input, version: 1 }))).status).toBe(
      404,
    );
    const own = await save({ ...input, rating: 2 });
    expect(own.shop.id).toBe(directory.shop.id);
    state.collector.mockResolvedValue({ id: "alice" });
    expect(
      (
        await DELETE(
          request("DELETE", { shopId: directory.shop.id, version: 1 }),
        )
      ).status,
    ).toBe(200);
    expect((await read()).shops).toEqual([osm]);
  });

  it("can clear rating and staff, and removes only the caller's saved shop", async () => {
    const first = await save();
    state.collector.mockResolvedValue({ id: "bob" });
    const bob = await save({ ...initial, rating: 1, employees: [] });
    state.collector.mockResolvedValue({ id: "alice" });
    const cleared = await save({
      ...initial,
      version: first.version,
      rating: null,
      employees: [],
    });
    expect(cleared).toMatchObject({ version: 2, rating: null, employees: [] });
    expect((await read()).shops).toEqual([cleared]);
    const removed = await DELETE(
      request("DELETE", { shopId: first.shop.id, version: cleared.version }),
    );
    expect(removed.status).toBe(200);
    expect(await removed.json()).toEqual({ ok: true });
    expect((await read()).shops).toEqual([]);
    state.collector.mockResolvedValue({ id: "bob" });
    expect((await read()).shops).toEqual([bob]);
  });

  it("allows only one concurrent create and never duplicates a shop", async () => {
    const responses = await Promise.all([
      PUT(request("PUT", initial)),
      PUT(request("PUT", initial)),
    ]);
    expect(responses.map((response) => response.status).sort()).toEqual([
      200, 409,
    ]);
    expect((await PUT(request("PUT", initial))).status).toBe(409);
    expect((await read()).shops).toHaveLength(1);
    expect((await read()).shops[0].version).toBe(1);
  });

  it("detects concurrent updates and stale deletes without losing staff edits", async () => {
    await save();
    const responses = await Promise.all([
      PUT(request("PUT", { ...initial, rating: 3, version: 1 })),
      PUT(request("PUT", { ...initial, rating: 4, version: 1 })),
    ]);
    expect(responses.map((response) => response.status).sort()).toEqual([
      200, 409,
    ]);
    const winner = await responses
      .find((response) => response.status === 200)!
      .json();
    expect((await read()).shops).toEqual([winner]);
    expect(
      (await DELETE(request("DELETE", { shopId: initial.shop.id, version: 1 })))
        .status,
    ).toBe(409);
    expect((await read()).shops).toEqual([winner]);
    expect(winner.version).toBe(2);
    expect(winner.employees).toHaveLength(1);
  });

  it("sorts saved shops by the latest personal edit", async () => {
    const first = await save();
    const second = await save({
      ...initial,
      shop: { ...initial.shop, id: "way/789", name: "Second Records" },
    });
    expect((await read()).shops.map((entry) => entry.shop.id)).toEqual([
      second.shop.id,
      first.shop.id,
    ]);
    const updated = await save({ ...initial, version: 1, rating: 4 });
    expect((await read()).shops).toEqual([updated, second]);
  });

  it("rejects invalid ratings, versions, avatar choices, duplicate IDs, and excess staff", async () => {
    const employee = initial.employees[0];
    const invalid = [
      ...[0, 6, 2.5, "5"].map((rating) => ({ ...initial, rating })),
      ...[0, -1, 1.5, "1"].map((version) => ({ ...initial, version })),
      { ...initial, employees: [{ ...employee, name: " " }] },
      { ...initial, employees: [{ ...employee, role: "x".repeat(81) }] },
      { ...initial, employees: [{ ...employee, avatar: "unknown" }] },
      { ...initial, employees: [{ ...employee, id: "not-a-uuid" }] },
      {
        ...initial,
        employees: [{ ...employee, privateEmail: "x@example.test" }],
      },
      {
        ...initial,
        employees: [employee, { ...employee, id: employee.id.toUpperCase() }],
      },
      { ...initial, employees: Array.from({ length: 31 }, () => employee) },
    ];
    for (const input of invalid)
      expect((await PUT(request("PUT", input))).status).toBe(400);
    expect((await read()).shops).toEqual([]);
    expect(
      saveShopSchema.parse({ ...initial, rating: null, employees: [] }).rating,
    ).toBeNull();
  });

  it("validates bounded shop identity, coordinates, text, and safe website URLs", () => {
    for (const id of [
      "node/0",
      "node/-1",
      "node/01",
      "node/1.2",
      "node/9007199254740992",
      "node/1?x=2",
      "user/1",
      "directory/",
      "directory/../shop",
      "directory/Shop",
      "directory/shop--name",
      "directory/" + "x".repeat(81),
    ])
      expect(notebookShopIdSchema.safeParse(id).success).toBe(false);
    for (const id of [
      "node/1",
      "way/123",
      "relation/9007199254740991",
      "directory/spoonful-records-columbus",
    ])
      expect(notebookShopIdSchema.safeParse(id).success).toBe(true);
    for (const patch of [
      { name: "" },
      { name: "x".repeat(241) },
      { address: "x".repeat(501) },
      { latitude: 91 },
      { longitude: -181 },
      { latitude: NaN },
      { longitude: Infinity },
      { website: "javascript:alert(1)" },
      { website: "data:text/html,hello" },
      { website: "https://user:password@example.test/" },
      { website: "https://example.test/\ninjected" },
      { website: "https://example.test/" + "x".repeat(2048) },
      { owner_id: "bob" },
    ])
      expect(
        saveShopSchema.safeParse({
          ...initial,
          shop: { ...initial.shop, ...patch },
        }).success,
      ).toBe(false);
    expect(
      saveShopSchema.safeParse({
        ...initial,
        shop: { ...initial.shop, address: null, website: null },
      }).success,
    ).toBe(true);
  });

  it("bounds actual streamed payload size and handles malformed JSON", async () => {
    const huge = new Request("https://wax.example.test/api/shops/notebook", {
      method: "PUT",
      headers: { Origin: "https://wax.example.test", "Content-Length": "2" },
      body: JSON.stringify({ ...initial, extra: "x".repeat(32 * 1024) }),
    });
    expect((await PUT(huge)).status).toBe(413);
    const malformed = new Request(
      "https://wax.example.test/api/shops/notebook",
      {
        method: "PUT",
        headers: { Origin: "https://wax.example.test" },
        body: "{bad-json",
      },
    );
    expect((await PUT(malformed)).status).toBe(400);
    expect((await read()).shops).toEqual([]);
  });

  it("cascades private shop notes when a collector is removed", async () => {
    await state.db!.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('temporary','Temporary','temporary@example.test'); INSERT INTO wax.collectors(id) VALUES('temporary')",
    );
    state.collector.mockResolvedValue({ id: "temporary" });
    await save();
    await state.db!.exec("DELETE FROM wax.users WHERE id='temporary'");
    expect(
      (
        await state.db!.query(
          "SELECT * FROM wax.shop_notebook WHERE owner_id='temporary'",
        )
      ).rows,
    ).toEqual([]);
  });
});
