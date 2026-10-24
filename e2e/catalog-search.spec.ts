import { expect, test, type Page } from "@playwright/test";
import {
  DEFAULT_COPY,
  EMPTY_CANDIDATE,
  type CatalogCandidate,
  type CreateRecordInput,
  type LibraryData,
  type RecordCopy,
  type UpdateRecordInput,
} from "../src/lib/contracts";
import type { CatalogAlbum, CatalogPage } from "../src/lib/catalog-search";

const copyId = "00000000-0000-4000-8000-000000000401";
const album: CatalogAlbum = {
  masterId: 46402,
  artist: "The Beatles",
  title: "The Beatles",
  originalYear: 1968,
  coverUrl: null,
  sourceUrl: "https://www.discogs.com/master/46402",
};
const candidate: CatalogCandidate = {
  discogsId: 4001,
  artist: "The Beatles",
  title: "The Beatles",
  originalYear: 1968,
  year: 1968,
  genres: ["Rock"],
  label: "Apple Records",
  catalogNumber: "SWBO 101",
  country: "US",
  format: "Vinyl, LP, Album",
  coverUrl: null,
  sourceUrl: "https://www.discogs.com/release/4001",
};
const pagination = (page = 1, pages = 1, total = 1): CatalogPage => ({
  page,
  pages,
  total,
  hasMore: page < pages,
});
function owned(overrides: Partial<RecordCopy> = {}): RecordCopy {
  return {
    ...DEFAULT_COPY,
    ...candidate,
    id: copyId,
    releaseId: copyId,
    version: 1,
    addedAt: "2026-10-24T12:00:00Z",
    updatedAt: "2026-10-24T12:00:00Z",
    crateIds: [],
    photoIds: [],
    spinCount: 0,
    lastPlayedAt: null,
    ...overrides,
  };
}
function library(records: RecordCopy[] = []): LibraryData {
  return {
    collector: {
      id: "00000000-0000-4000-8000-000000000402",
      name: "Catalog browser fixture",
      email: "catalog@example.test",
      isOwner: false,
    },
    records,
    crates: [],
    sharedCrates: [],
    listens: [],
  };
}
async function mockCatalog(page: Page, data: LibraryData) {
  const requests: URL[] = [];
  let details = 0;
  await page.route("**/api/library", (route) => route.fulfill({ json: data }));
  await page.route("**/api/catalog/albums?**", (route) => {
    requests.push(new URL(route.request().url()));
    return route.fulfill({
      json: {
        albums: [album],
        pagination: pagination(),
        notice: "The White Album is cataloged as The Beatles.",
      },
    });
  });
  await page.route("**/api/catalog/albums/46402/versions?**", (route) => {
    const url = new URL(route.request().url());
    requests.push(url);
    const pageNumber = Number(url.searchParams.get("page") || 1);
    const found =
      pageNumber === 2
        ? { ...candidate, discogsId: 4002, catalogNumber: "SWBO 101 second" }
        : candidate;
    return route.fulfill({
      json: {
        candidates: [found],
        pagination: pagination(pageNumber, 2, 24),
      },
    });
  });
  await page.route(/\/api\/catalog\/400[12]$/, (route) => {
    details++;
    return route.fulfill({
      json: {
        ...candidate,
        discogsId: Number(route.request().url().split("/").at(-1)),
      },
    });
  });
  return { requests, details: () => details };
}

test("album discovery narrows and paginates pressings before fetching canonical details for review", async ({
  page,
}) => {
  const data = library();
  const catalog = await mockCatalog(page, data);
  const writes: CreateRecordInput[] = [];
  await page.route("**/api/records", (route) => {
    const body = route.request().postDataJSON() as CreateRecordInput;
    writes.push(body);
    const record = owned({ ...body.candidate, ...body.copy });
    data.records = [record];
    return route.fulfill({ json: record });
  });
  await page.goto("/add");
  await page
    .getByLabel("Search the record catalog")
    .fill("beatles white album");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByText("The White Album is cataloged as The Beatles."),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "See pressings", exact: true })
    .click();
  await page
    .getByRole("combobox", { name: /^Pressing country/ })
    .selectOption("US");
  await page
    .getByLabel("Label / company", { exact: true })
    .fill("Apple Records");
  await page.getByLabel("Pressing year", { exact: true }).fill("1968");
  await page
    .getByRole("button", { name: "Apply filters", exact: true })
    .click();
  await expect
    .poll(() =>
      catalog.requests.some(
        (url) =>
          url.searchParams.get("country") === "US" &&
          url.searchParams.get("year") === "1968" &&
          url.searchParams.get("label") === "Apple Records",
      ),
    )
    .toBeTruthy();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: test.info().outputPath("filtered-pressings.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Next results", exact: true }).click();
  await expect(page.getByText("SWBO 101 second")).toBeVisible();
  expect(catalog.requests.at(-1)?.searchParams.get("page")).toBe("2");
  expect(catalog.requests.at(-1)?.searchParams.get("country")).toBe("US");
  expect(writes).toHaveLength(0);
  await page
    .getByRole("button", { name: "Review this pressing", exact: true })
    .click();
  await expect(page.getByLabel("Record title", { exact: true })).toHaveValue(
    "The Beatles",
  );
  await expect(page.getByLabel("Catalog number", { exact: true })).toHaveValue(
    "SWBO 101",
  );
  await expect(
    page.getByLabel("Album release year", { exact: false }),
  ).toHaveValue("1968");
  await expect(
    page.getByRole("checkbox", { name: /I’ve checked this pressing/ }),
  ).not.toBeChecked();
  expect(catalog.details()).toBe(1);
  expect(writes).toHaveLength(0);
  await page
    .getByLabel("Your liner notes", { exact: false })
    .fill("My copy, still checking the label.");
  await page
    .getByRole("button", { name: "Add to my collection", exact: true })
    .click();
  await expect(page).toHaveURL(new RegExp(`/records/${copyId}$`));
  expect(writes).toHaveLength(1);
  expect(writes[0].candidate).toMatchObject({
    discogsId: 4002,
    originalYear: 1968,
    catalogNumber: "SWBO 101",
  });
  expect(writes[0].copy).toMatchObject({
    notes: "My copy, still checking the label.",
    pressingConfirmed: false,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
});

test("existing handwritten records can review catalog details without losing private copy fields or saving early", async ({
  page,
}) => {
  const data = library([
    owned({
      ...EMPTY_CANDIDATE,
      artist: "Beatles",
      title: "White Album",
      notes: "Bought with Dad",
      purchasePrice: "27.50",
      currency: "USD",
      location: "Living room A",
      rating: 5,
      mediaCondition: "VG+",
      sleeveCondition: "VG",
      intent: "keep",
      acquiredAt: "2026-10-01",
      pressingConfirmed: true,
    }),
  ]);
  await mockCatalog(page, data);
  const patches: UpdateRecordInput[] = [];
  await page.route(`**/api/records/${copyId}`, (route) => {
    const body = route.request().postDataJSON() as UpdateRecordInput;
    patches.push(body);
    data.records[0] = {
      ...data.records[0],
      ...body,
      ...body.candidate,
      version: 2,
    };
    return route.fulfill({ json: data.records[0] });
  });
  await page.goto(`/records/${copyId}`);
  await page
    .getByRole("button", { name: "Edit this copy", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Find catalog matches", exact: true })
    .click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog
    .getByRole("button", { name: "See pressings", exact: true })
    .click();
  await page.screenshot({
    path: test.info().outputPath("enrichment-dialog.png"),
  });
  await dialog
    .getByRole("button", { name: "Review this pressing", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByLabel("Record title", { exact: true })).toHaveValue(
    "The Beatles",
  );
  await expect(
    page.getByLabel("Your liner notes", { exact: false }),
  ).toHaveValue("Bought with Dad");
  await expect(page.getByLabel("Purchase price", { exact: true })).toHaveValue(
    "27.50",
  );
  await expect(
    page.getByLabel("Shelf / location", { exact: true }),
  ).toHaveValue("Living room A");
  await expect(
    page.getByRole("combobox", { name: /^Media condition/ }),
  ).toHaveValue("VG+");
  await expect(
    page.getByRole("checkbox", { name: /I’ve checked this pressing/ }),
  ).not.toBeChecked();
  expect(patches).toHaveLength(0);
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(
    page.getByText("Your copy is up to date.", { exact: true }),
  ).toBeVisible();
  expect(patches).toHaveLength(1);
  expect(patches[0]).toMatchObject({
    version: 1,
    candidate: { discogsId: 4001, originalYear: 1968 },
    notes: "Bought with Dad",
    purchasePrice: "27.50",
    location: "Living room A",
    mediaCondition: "VG+",
    sleeveCondition: "VG",
    rating: 5,
    pressingConfirmed: false,
    acquiredAt: "2026-10-01",
  });
});

test("typing a manual album finds catalog suggestions while retaining notes and purchase details", async ({
  page,
}) => {
  const data = library();
  const catalog = await mockCatalog(page, data);
  let saves = 0;
  await page.route("**/api/records", (route) => {
    saves++;
    return route.fulfill({
      status: 500,
      json: { error: "This test never saves." },
    });
  });
  await page.goto("/add?mode=manual");
  await page
    .getByRole("button", { name: "Enter the details", exact: true })
    .click();
  await page.getByLabel("Artist", { exact: true }).fill("Beatles");
  await page.getByLabel("Record title", { exact: true }).fill("White Album");
  await page
    .getByLabel("Your liner notes", { exact: false })
    .fill("A gift from my friend.");
  await page.getByLabel("Purchase price", { exact: true }).fill("35.75");
  const suggestions = page.getByRole("region", { name: "Catalog suggestions" });
  await expect(
    suggestions.getByRole("button").filter({ hasText: "See pressings" }),
  ).toBeVisible();
  expect(
    catalog.requests.some(
      (url) =>
        url.searchParams.get("artist") === "beatles" &&
        url.searchParams.get("title") === "white album",
    ),
  ).toBeTruthy();
  await expect(page.getByLabel("Record title", { exact: true })).toHaveValue(
    "White Album",
  );
  await suggestions
    .getByRole("button")
    .filter({ hasText: "See pressings" })
    .click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog
    .getByRole("button", { name: "Review this pressing", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByLabel("Record title", { exact: true })).toHaveValue(
    "The Beatles",
  );
  await expect(
    page.getByLabel("Your liner notes", { exact: false }),
  ).toHaveValue("A gift from my friend.");
  await expect(page.getByLabel("Purchase price", { exact: true })).toHaveValue(
    "35.75",
  );
  await expect(
    page.getByRole("checkbox", { name: /I’ve checked this pressing/ }),
  ).not.toBeChecked();
  expect(catalog.details()).toBe(1);
  expect(saves).toBe(0);
});

test("throttled pressing filters preserve the album, controls, and previous results for a retry", async ({
  page,
}) => {
  await mockCatalog(page, library());
  await page.goto("/add?artist=The%20Beatles&title=White%20Album");
  await page
    .getByRole("button", { name: "See pressings", exact: true })
    .click();
  let attempts = 0;
  const attemptedFilters: URL[] = [];
  await page.route("**/api/catalog/albums/46402/versions?**", (route) => {
    attemptedFilters.push(new URL(route.request().url()));
    attempts++;
    return route.fulfill(
      attempts === 1
        ? {
            status: 429,
            json: { error: "Catalog is busy. Please try again shortly." },
          }
        : { json: { candidates: [candidate], pagination: pagination() } },
    );
  });
  await page
    .getByRole("combobox", { name: /^Pressing country/ })
    .selectOption("US");
  await page
    .getByLabel("Label / company", { exact: true })
    .fill("Apple Records");
  await page.getByLabel("Pressing year", { exact: true }).fill("1968");
  await page
    .getByRole("button", { name: "Apply filters", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("Catalog is busy");
  await expect(
    page.getByRole("combobox", { name: /^Pressing country/ }),
  ).toHaveValue("US");
  await expect(page.getByLabel("Label / company", { exact: true })).toHaveValue(
    "Apple Records",
  );
  await expect(page.getByLabel("Pressing year", { exact: true })).toHaveValue(
    "1968",
  );
  await expect(
    page.getByRole("button", { name: "Review this pressing", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Try catalog again", exact: true })
    .click();
  await expect(page.getByRole("alert")).not.toBeVisible();
  expect(attempts).toBe(2);
  expect(attemptedFilters[0].search).toBe(attemptedFilters[1].search);
  await expect(page.getByLabel("Search artist", { exact: true })).toHaveValue(
    "The Beatles",
  );
  await expect(
    page.getByLabel("Search album title", { exact: true }),
  ).toHaveValue("White Album");
});

test("photo suggestions can be refined into another country's pressing without losing the uploaded photo", async ({
  page,
}) => {
  const data = library();
  await mockCatalog(page, data);
  const photoId = "00000000-0000-4000-8000-000000000403";
  let uploads = 0;
  let saved: CreateRecordInput | undefined;
  await page.route("**/api/photos", (route) => {
    uploads++;
    return route.fulfill({
      json: { id: photoId, url: `/api/photos/${photoId}` },
    });
  });
  await page.route("**/api/identify", (route) =>
    route.fulfill({
      json: {
        photoId,
        clues: {
          artist: "The Beatles",
          title: "The Beatles",
          label: null,
          barcode: null,
          catalogNumber: null,
          uncertainty: "Cover only; pressing is not established.",
        },
        candidates: [
          { ...candidate, discogsId: 3999, country: "Russia", year: 1991 },
        ],
      },
    }),
  );
  await page.route("**/api/records", (route) => {
    saved = route.request().postDataJSON() as CreateRecordInput;
    const record = owned({
      ...saved.candidate,
      ...saved.copy,
      photoIds: saved.photoIds,
    });
    data.records = [record];
    return route.fulfill({ json: record });
  });
  await page.goto("/add?mode=photo");
  await page.locator('input[type="file"]').setInputFiles({
    name: "cover-fixture.png",
    mimeType: "image/png",
    buffer: await page.screenshot(),
  });
  await page
    .getByRole("button", { name: "Identify this record", exact: true })
    .click();
  await expect(
    page.getByText("Photo clues, ready to refine.", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Russia", { exact: true })).toBeVisible();
  await page
    .getByLabel("Search album title", { exact: true })
    .fill("White Album");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page
    .getByRole("button", { name: "See pressings", exact: true })
    .click();
  await page
    .getByRole("combobox", { name: /^Pressing country/ })
    .selectOption("US");
  await page
    .getByRole("button", { name: "Apply filters", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Review this pressing", exact: true })
    .click();
  await expect(page.getByLabel("Country", { exact: true })).toHaveValue("US");
  await expect(
    page.getByRole("checkbox", { name: /I’ve checked this pressing/ }),
  ).not.toBeChecked();
  await page
    .getByRole("button", { name: "Add to my collection", exact: true })
    .click();
  await expect(page).toHaveURL(new RegExp(`/records/${copyId}$`));
  expect(uploads).toBe(1);
  expect(saved?.photoIds).toEqual([photoId]);
  expect(saved?.candidate).toMatchObject({ discogsId: 4001, country: "US" });
});
