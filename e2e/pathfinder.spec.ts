import { expect, test, type Page } from "@playwright/test";
import { DEFAULT_COPY, type LibraryData } from "../src/lib/contracts";
import type { NearbyShops, RecordShop, ShopRadiusKm } from "../src/lib/shops";

const recordId = "00000000-0000-4000-8000-000000000102";
const library: LibraryData = {
  collector: {
    id: "00000000-0000-4000-8000-000000000001",
    name: "Pathfinder fixture",
    email: "pathfinder@example.test",
    isOwner: false,
  },
  records: [
    {
      ...DEFAULT_COPY,
      id: recordId,
      releaseId: recordId,
      artist: "The Beach Boys",
      title: "Pet Sounds",
      year: 2024,
      genres: ["Rock", "Pop Rock", "Psychedelic Rock"],
      label: "Capitol",
      catalogNumber: "ST 2458",
      country: "US",
      format: "Vinyl, LP, Album, Reissue, Repress, Stereo",
      coverUrl: null,
      discogsId: 12345,
      sourceUrl: "https://www.discogs.com/release/12345",
      version: 1,
      addedAt: "2026-10-23T12:00:00Z",
      updatedAt: "2026-10-23T12:00:00Z",
      crateIds: [],
      spinCount: 0,
      lastPlayedAt: null,
      photoIds: [],
    },
  ],
  crates: [],
  sharedCrates: [],
  listens: [],
};
const center = { latitude: 40.02, longitude: -83.04 };
const shops: RecordShop[] = [
  ["Local Groove", 3, 40.04, -83.04],
  ["Soul Cellar", 7, 40.08, -83.04],
  ["Outer Crate", 18, 40.18, -83.04],
  ["County Line Vinyl", 40, 40.38, -83.04],
  ["Day Trip Records", 80, 40.74, -83.04],
].map(([name, distanceKm, latitude, longitude], index) => ({
  id: `directory/fixture-${index + 1}`,
  name: String(name),
  distanceKm: Number(distanceKm),
  latitude: Number(latitude),
  longitude: Number(longitude),
  address: `${index + 1} Main Street, Ohio`,
  website: "https://example.com",
  phone: null,
  openingHours: null,
  osmUrl: null,
  sourceUrl: "https://example.com",
  sourceLabel: "Retailer website",
}));

async function fixture(page: Page, sameResults = false, partial = false) {
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
  await page.route("**/api/shops/notebook", (route) =>
    route.fulfill({ json: { shops: [] } }),
  );
  await page.route("**/api/shops/places", (route) =>
    route.fulfill({
      json: {
        places: [
          { id: "columbus", label: "Columbus, Ohio", ...center },
          {
            id: "other",
            label: "Other city, Ohio",
            latitude: 41.1,
            longitude: -83.1,
          },
        ],
      },
    }),
  );
  await page.route("**/api/shops/nearby", (route) => {
    const input = route.request().postDataJSON() as typeof center & {
      radiusKm: ShopRadiusKm;
    };
    const result: NearbyShops = {
      center: { latitude: input.latitude, longitude: input.longitude },
      radiusKm: input.radiusKm,
      fetchedAt: "2026-10-23T12:00:00Z",
      limited: false,
      discovery: { mapAvailable: !partial, directoryCount: 2 },
      shops: shops.filter(
        (shop) => shop.distanceKm <= (sameResults ? 10 : input.radiusKm),
      ),
    };
    return route.fulfill({ json: result });
  });
  await page.goto("/shops");
  await page.getByLabel("Or enter a city or postal code").fill("Columbus");
  await page.getByRole("button", { name: "Find area", exact: true }).click();
  await page
    .getByRole("button", { name: "Columbus, Ohio", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "2 places to dig", exact: true }),
  ).toBeVisible();
}

test("radius expansion finds farther shops and the four-stop route survives filtering and widening", async ({
  page,
}) => {
  await fixture(page);
  await page.getByRole("radio", { name: "15 miles", exact: true }).check();
  await expect(
    page.getByRole("heading", { name: "3 places to dig", exact: true }),
  ).toBeVisible();
  await page.getByRole("radio", { name: "30 miles", exact: true }).check();
  await expect(
    page.getByRole("heading", { name: "4 places to dig", exact: true }),
  ).toBeVisible();
  for (const shop of shops.slice(0, 4)) {
    await page
      .locator(".shop-card")
      .filter({
        has: page.getByRole("heading", { name: shop.name, exact: true }),
      })
      .getByRole("button", { name: "Add to dig route", exact: true })
      .click();
  }
  const route = page.getByRole("region", {
    name: "Your dig route",
    exact: true,
  });
  await route
    .getByRole("button", {
      name: "Move County Line Vinyl earlier",
      exact: true,
    })
    .click();
  const url = new URL(
    (await route
      .getByRole("link", { name: "Open dig route", exact: true })
      .getAttribute("href"))!,
  );
  expect(url.searchParams.get("waypoints")).toBe(
    "40.04,-83.04|40.08,-83.04|40.38,-83.04",
  );
  expect(url.searchParams.get("destination")).toBe("40.18,-83.04");
  expect(url.searchParams.get("origin")).toBe("40.02,-83.04");
  await page.getByRole("radio", { name: /60 miles/ }).check();
  await expect(
    page.getByRole("heading", { name: "5 places to dig", exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".shop-card")
      .filter({
        has: page.getByRole("heading", {
          name: "Day Trip Records",
          exact: true,
        }),
      })
      .getByRole("button", { name: "Add to dig route", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Filter these shops", { exact: true }).fill("County");
  await expect(page.locator(".shop-card h3")).toHaveText(["County Line Vinyl"]);
  await expect(
    route.getByRole("link", { name: "Open dig route", exact: true }),
  ).toHaveAttribute("href", url.href);
  await route.getByRole("button", { name: "Clear route", exact: true }).click();
  await expect(
    route.getByRole("link", { name: "Open dig route", exact: true }),
  ).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: `.wax-local/pathfinder-${test.info().project.name}.png`,
    fullPage: true,
  });
});

test("unchanged expanded results explain coverage and moving city clears the old trip", async ({
  page,
}) => {
  await fixture(page, true, true);
  await expect(
    page.getByText("Map search is taking a break.", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".shop-card")
    .first()
    .getByRole("button", { name: "Add to dig route", exact: true })
    .click();
  await page.getByRole("radio", { name: "30 miles", exact: true }).check();
  await expect(page.getByText(/no additional/i)).toBeVisible();
  await page.getByLabel("Or enter a city or postal code").fill("Other");
  await page.getByRole("button", { name: "Find area", exact: true }).click();
  await page
    .getByRole("button", { name: "Other city, Ohio", exact: true })
    .click();
  await expect(
    page
      .getByRole("region", { name: "Your dig route", exact: true })
      .getByRole("link", { name: "Open dig route", exact: true }),
  ).toHaveCount(0);
});

test("Artist Tree stays below the spin and QR actions and above the attribution without overlap", async ({
  page,
}) => {
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
  await page.goto(`/records/${recordId}`);
  const tree = page.getByRole("link", { name: "Artist Tree", exact: true });
  await expect(tree).toBeVisible();
  const widths =
    test.info().project.name === "desktop-chromium"
      ? [1440, 1141, 1000, 768, 390]
      : [page.viewportSize()!.width];
  for (const width of widths) {
    await page.setViewportSize({ width, height: 1000 });
    await tree.scrollIntoViewIfNeeded();
    const box = (await tree.boundingBox())!;
    const spin = (await page
      .getByRole("button", { name: "Log a spin", exact: true })
      .boundingBox())!;
    const qr = (await page
      .getByRole("button", { name: "QR label", exact: true })
      .boundingBox())!;
    const source = (await page
      .getByRole("link", { name: /Data provided by Discogs/ })
      .boundingBox())!;
    expect(box.y).toBeGreaterThanOrEqual(
      Math.max(spin.y + spin.height, qr.y + qr.height),
    );
    expect(source.y).toBeGreaterThanOrEqual(box.y + box.height);
    expect(box.width).toBeGreaterThanOrEqual(spin.width);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBeTruthy();
  }
  await page.screenshot({
    path: `.wax-local/record-actions-${test.info().project.name}.png`,
    fullPage: true,
  });
});
