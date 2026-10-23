import { expect, test } from "@playwright/test";
import {
  DEFAULT_COPY,
  type CatalogCandidate,
  type LibraryData,
  type RecordCopy,
} from "../src/lib/contracts";

const copyId = "00000000-0000-4000-8000-000000000301";

function record(overrides: Partial<RecordCopy> = {}): RecordCopy {
  return {
    ...DEFAULT_COPY,
    id: copyId,
    releaseId: copyId,
    discogsId: 12345,
    artist: "The Beach Boys",
    title: "Pet Sounds",
    originalYear: 1966,
    year: 2024,
    genres: ["Pop"],
    label: "Capitol",
    catalogNumber: "ST 2458",
    country: "US",
    format: "Vinyl, LP, Reissue",
    coverUrl: null,
    sourceUrl: "https://www.discogs.com/release/12345",
    version: 1,
    addedAt: "2026-10-23T12:00:00Z",
    updatedAt: "2026-10-23T12:00:00Z",
    crateIds: [],
    spinCount: 0,
    lastPlayedAt: null,
    photoIds: [],
    ...overrides,
  };
}

function library(records: RecordCopy[]): LibraryData {
  return {
    collector: {
      id: "00000000-0000-4000-8000-000000000302",
      name: "Original year fixture",
      email: "year@example.test",
      isOwner: false,
    },
    records,
    crates: [],
    sharedCrates: [],
    listens: [],
  };
}

test("the collection shows the album year while details and editing retain the pressing year", async ({
  page,
}) => {
  const data = library([record()]);
  let savedCandidate: CatalogCandidate | undefined;
  await page.route("**/api/library", (route) => route.fulfill({ json: data }));
  await page.route(`**/api/records/${copyId}`, (route) => {
    expect(route.request().method()).toBe("PATCH");
    const body = route.request().postDataJSON() as {
      candidate: CatalogCandidate;
    };
    savedCandidate = body.candidate;
    data.records[0] = { ...data.records[0], ...body.candidate, version: 2 };
    return route.fulfill({ json: data.records[0] });
  });

  await page.goto("/collection");
  const facts = page.locator(".record-card .record-facts");
  await expect(facts).toContainText("1966");
  await expect(facts).not.toContainText("2024");
  await page.getByRole("button", { name: "Table view", exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "ALBUM YEAR / GENRE" }),
  ).toBeVisible();
  await expect(page.locator(".record-table tbody tr td").nth(1)).toContainText(
    "1966",
  );
  await expect(page.locator(".record-table tbody")).not.toContainText("2024");

  await page.goto(`/records/${copyId}`);
  await expect(
    page.locator(".detail-facts > div").filter({
      has: page.locator("dt", { hasText: /^Originally released$/ }),
    }),
  ).toContainText("1966");
  await expect(
    page
      .locator(".detail-facts > div")
      .filter({ has: page.locator("dt", { hasText: /^Pressing year$/ }) }),
  ).toContainText("2024");
  await page
    .getByRole("button", { name: "Edit this copy", exact: true })
    .click();
  await expect(
    page.getByLabel("Album release year", { exact: false }),
  ).toHaveValue("1966");
  await expect(page.getByLabel("Pressing year", { exact: false })).toHaveValue(
    "2024",
  );
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(
    page.getByText("Your copy is up to date.", { exact: true }),
  ).toBeVisible();
  expect(savedCandidate).toMatchObject({ originalYear: 1966, year: 2024 });
});

test("album year sorting ignores reissue dates and keeps unknown original years last", async ({
  page,
}) => {
  const data = library([
    record(),
    record({
      id: "00000000-0000-4000-8000-000000000303",
      title: "Later Album",
      originalYear: 1971,
      year: 1980,
    }),
    record({
      id: "00000000-0000-4000-8000-000000000304",
      title: "Unknown Album",
      originalYear: null,
      year: 1955,
    }),
  ]);
  await page.route("**/api/library", (route) => route.fulfill({ json: data }));
  await page.goto("/collection");
  await expect(
    page
      .locator(".record-card")
      .filter({ hasText: "Unknown Album" })
      .locator(".record-facts"),
  ).toContainText("Year unknown");
  await expect(
    page
      .locator(".record-card")
      .filter({ hasText: "Unknown Album" })
      .locator(".record-facts"),
  ).not.toContainText("1955");
  await page
    .getByRole("combobox", { name: "Sort records", exact: true })
    .selectOption("year-oldest");
  await expect(page.locator(".record-caption h3")).toHaveText([
    "Pet Sounds",
    "Later Album",
    "Unknown Album",
  ]);
  await page
    .getByRole("combobox", { name: "Sort records", exact: true })
    .selectOption("year-newest");
  await expect(page.locator(".record-caption h3")).toHaveText([
    "Later Album",
    "Pet Sounds",
    "Unknown Album",
  ]);
});

test("legacy album years fill after loading without replacing the pressing year or repeating the write", async ({
  page,
}) => {
  const data = library([record({ originalYear: undefined })]);
  data.needsOriginalYears = true;
  let calls = 0;
  await page.route("**/api/library", (route) => route.fulfill({ json: data }));
  await page.route("**/api/records/years", (route) => {
    calls++;
    expect(route.request().method()).toBe("POST");
    expect(route.request().postDataJSON()).toEqual({});
    data.records[0].originalYear = 1966;
    data.records[0].version = 2;
    data.needsOriginalYears = false;
    return route.fulfill({
      json: { updated: 1, processed: 1, remaining: 0, transientFailure: false },
    });
  });
  await page.goto("/collection");
  await expect(page.locator(".record-card .record-facts")).toContainText(
    "1966",
  );
  await expect(page.locator(".record-card .record-facts")).not.toContainText(
    "2024",
  );
  expect(data.records[0].year).toBe(2024);
  expect(calls).toBe(1);
  await page.reload();
  await expect(page.locator(".record-card .record-facts")).toContainText(
    "1966",
  );
  expect(calls).toBe(1);
});

test("a year-provider outage leaves the collection visible without a request loop", async ({
  page,
}) => {
  const data = library([record({ originalYear: undefined })]);
  data.needsOriginalYears = true;
  let calls = 0;
  await page.route("**/api/library", (route) => route.fulfill({ json: data }));
  await page.route("**/api/records/years", (route) => {
    calls++;
    return route.fulfill({
      status: 503,
      json: { error: "Catalog temporarily unavailable" },
    });
  });
  await page.goto("/collection");
  await expect(page.locator(".record-card .record-facts")).toContainText(
    "Year unknown",
  );
  await expect(page.locator(".record-card .record-facts")).not.toContainText(
    "2024",
  );
  await page.getByRole("button", { name: "Table view", exact: true }).click();
  await expect(page.locator(".record-table tbody")).toContainText("Pet Sounds");
  expect(calls).toBe(1);
});
