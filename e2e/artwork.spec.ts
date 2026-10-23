import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  DEFAULT_COPY,
  type LibraryData,
  type RecordCopy,
} from "../src/lib/contracts";

const recordId = "00000000-0000-4000-8000-000000000010";
const artworkPath = "/api/catalog/12345/artwork";
const sleeveLabel = "Curtis Mayfield — Super Fly sleeve";
// A small valid PNG keeps this regression independent of external image hosts.
const image = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0X0AAAAASUVORK5CYII=",
  "base64",
);

function fixture(): LibraryData {
  const record: RecordCopy = {
    ...DEFAULT_COPY,
    id: recordId,
    releaseId: "00000000-0000-4000-8000-000000000011",
    discogsId: 12345,
    artist: "Curtis Mayfield",
    title: "Super Fly",
    year: 1972,
    genres: ["Soul"],
    label: "Curtom",
    catalogNumber: "CRS 8014",
    country: "US",
    format: "Vinyl, LP",
    // Linked copies keep this stable path when provider metadata expires.
    coverUrl: artworkPath,
    sourceUrl: "https://www.discogs.com/release/12345",
    version: 1,
    addedAt: "2026-10-21T12:00:00.000Z",
    updatedAt: "2026-10-21T12:00:00.000Z",
    crateIds: [],
    spinCount: 1,
    lastPlayedAt: "2026-10-22T12:00:00.000Z",
    photoIds: [],
  };
  return {
    collector: {
      id: "00000000-0000-4000-8000-000000000001",
      name: "WAX artwork fixture",
      email: "artwork-browser@example.com",
      isOwner: false,
    },
    records: [record],
    crates: [],
    sharedCrates: [
      {
        id: "00000000-0000-4000-8000-000000000012",
        name: "Shared soul favorites",
        description: "A browser fixture crate",
        color: "#bc4825",
        ownerName: "Another collector",
        records: [record],
      },
    ],
    listens: [
      {
        id: "00000000-0000-4000-8000-000000000013",
        recordId,
        artist: record.artist,
        title: record.title,
        coverUrl: artworkPath,
        playedAt: "2026-10-22T12:00:00.000Z",
        notes: "A good first spin.",
      },
    ],
  };
}

async function mockLibrary(page: Page, library = fixture()) {
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
}

async function expectLoaded(cover: Locator) {
  await expect(cover).toBeVisible();
  await cover.scrollIntoViewIfNeeded();
  await expect
    .poll(async () =>
      cover.evaluate(
        (element) =>
          element instanceof HTMLImageElement &&
          element.complete &&
          element.naturalWidth > 0,
      ),
    )
    .toBe(true);
  expect(
    new URL((await cover.getAttribute("src"))!, "https://wax.example").pathname,
  ).toBe(artworkPath);
}

test("linked artwork loads on the collection, record, shared crate, and listening journal", async ({
  page,
}) => {
  await mockLibrary(page);
  let artworkRequests = 0;
  await page.route("**/api/catalog/12345/artwork*", (route) => {
    artworkRequests++;
    return route.fulfill({
      status: 200,
      contentType: "image/png",
      headers: { "cache-control": "private, no-store" },
      body: image,
    });
  });
  await page.goto("/collection");
  await expectLoaded(
    page.locator(".record-card").getByRole("img", { name: sleeveLabel }),
  );
  await page.goto(`/records/${recordId}`);
  await expectLoaded(
    page.locator(".detail-art").getByRole("img", { name: sleeveLabel }),
  );
  await page.goto("/crates");
  await expectLoaded(
    page
      .locator("#shared .crate-previews")
      .getByRole("img", { name: sleeveLabel }),
  );
  await page.getByRole("button", { name: "Take a look" }).click();
  await expectLoaded(
    page
      .getByRole("dialog", { name: "Shared soul favorites" })
      .getByRole("img", { name: sleeveLabel }),
  );
  await page.getByRole("button", { name: "Close dialog" }).click();
  await page.goto("/listening");
  await expectLoaded(
    page.locator(".listen-row").getByRole("img", { name: sleeveLabel }),
  );
  expect(artworkRequests).toBeGreaterThanOrEqual(4);
});

test("a manual record without artwork keeps its readable fallback without a catalog request", async ({
  page,
}) => {
  const library = fixture();
  library.records[0] = {
    ...library.records[0],
    discogsId: null,
    coverUrl: null,
    sourceUrl: null,
  };
  library.sharedCrates = [];
  library.listens = [];
  await mockLibrary(page, library);
  const artworkRequests: string[] = [];
  await page.route("**/api/catalog/**/artwork*", (route) => {
    artworkRequests.push(route.request().url());
    return route.fulfill({ status: 404 });
  });
  await page.goto("/collection");
  const fallback = page.locator(".record-card .cover-fallback");
  await expect(fallback).toContainText("Super Fly");
  await expect(fallback).toContainText("Curtis Mayfield");
  await expect(page.locator(".record-card img")).toHaveCount(0);
  expect(artworkRequests).toEqual([]);
});

test("a temporary artwork failure automatically retries and restores the sleeve", async ({
  page,
}) => {
  await mockLibrary(page);
  let artworkRequests = 0;
  await page.route("**/api/catalog/12345/artwork*", (route) => {
    artworkRequests++;
    return artworkRequests === 1
      ? route.fulfill({ status: 503, body: "Artwork temporarily unavailable" })
      : route.fulfill({ contentType: "image/png", body: image });
  });
  await page.goto("/collection");
  await expect(page.locator(".record-card .cover-fallback")).toContainText(
    "Super Fly",
  );
  await expectLoaded(
    page.locator(".record-card").getByRole("img", { name: sleeveLabel }),
  );
  expect(artworkRequests).toBe(2);
  await expect(page).toHaveURL(/\/collection$/);
});

test("persistent artwork failures keep a readable fallback and recover when connectivity returns", async ({
  page,
}) => {
  await mockLibrary(page);
  let available = false;
  let artworkRequests = 0;
  await page.route("**/api/catalog/12345/artwork*", (route) => {
    artworkRequests++;
    return available
      ? route.fulfill({ contentType: "image/png", body: image })
      : route.fulfill({ status: 503, body: "Artwork temporarily unavailable" });
  });
  await page.goto(`/records/${recordId}`);
  await expect.poll(() => artworkRequests).toBe(2);
  await expect(page.locator(".detail-art .cover-fallback")).toContainText(
    "Super Fly",
  );
  // Wait through one extra retry interval to catch an accidental endless loop.
  await page.waitForTimeout(2500);
  expect(artworkRequests).toBe(2);
  available = true;
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await expectLoaded(
    page.locator(".detail-art").getByRole("img", { name: sleeveLabel }),
  );
  expect(artworkRequests).toBe(3);
});
