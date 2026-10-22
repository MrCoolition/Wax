import { expect, test, type Page } from "@playwright/test";
import type { LibraryData } from "../src/lib/contracts";
import type { NearbyShops, ShopPlace, ShopRadiusKm } from "../src/lib/shops";

const library: LibraryData = {
  collector: {
    id: "00000000-0000-4000-8000-000000000001",
    name: "WAX browser fixture",
    email: "shops-browser@example.com",
    isOwner: false,
  },
  records: [],
  crates: [],
  sharedCrates: [],
  listens: [],
};
const places: ShopPlace[] = [
  {
    id: "new-york",
    label: "Brooklyn, New York, United States",
    latitude: 40.678178,
    longitude: -73.944158,
  },
  {
    id: "connecticut",
    label: "Brooklyn, Connecticut, United States",
    latitude: 41.788178,
    longitude: -71.949269,
  },
];

function nearby(radiusKm: ShopRadiusKm = 10): NearbyShops {
  return {
    center: { latitude: 40.71, longitude: -74.01 },
    radiusKm,
    fetchedAt: "2026-10-22T12:00:00.000Z",
    limited: false,
    // Intentionally out of distance order: the UI must put the closest first.
    shops: [
      {
        id: "node/103",
        name: "Long Play Records",
        distanceKm: 4.2,
        address: "56 Atlantic Avenue, Brooklyn, New York",
        website: null,
        phone: null,
        openingHours: null,
        osmUrl: "https://www.openstreetmap.org/node/103",
        latitude: 40.68,
        longitude: -73.99,
      },
      {
        id: "node/101",
        name: "The Soul Crate",
        distanceKm: 0.7,
        address: "123 Very Long Record Collector Avenue, New York, NY 10013",
        website: "https://example.com/record-shop",
        phone: "+1 212 555 0100",
        openingHours: "Mo-Sa 11:00-19:00; Su 12:00-17:00",
        osmUrl: "https://www.openstreetmap.org/node/101",
        latitude: 40.71642,
        longitude: -74.00218,
      },
      {
        id: "node/102",
        name: "Needle & Groove",
        distanceKm: 2.1,
        address: null,
        website: null,
        phone: null,
        openingHours: null,
        osmUrl: "https://www.openstreetmap.org/node/102",
        latitude: 40.72,
        longitude: -73.99,
      },
    ],
  };
}

async function mockLibrary(page: Page) {
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
}

async function expectNoOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
}

test("location is opt-in, rounded in a POST, and produces ordered shop directions", async ({
  page,
  context,
  baseURL,
}) => {
  await mockLibrary(page);
  // Keep the native browser API in this scenario so restrictive Permissions-Policy
  // headers cause a real failure instead of being hidden by a geolocation stub.
  await context.grantPermissions(["geolocation"], {
    origin: new URL(baseURL!).origin,
  });
  await context.setGeolocation({ latitude: 40.71427, longitude: -74.00597 });
  await page.addInitScript(() => {
    const getCurrentPosition = navigator.geolocation.getCurrentPosition.bind(
      navigator.geolocation,
    );
    const state = window as typeof window & { locationRequests: number };
    state.locationRequests = 0;
    navigator.geolocation.getCurrentPosition = (...args) => {
      state.locationRequests++;
      getCurrentPosition(...args);
    };
  });
  const bodies: unknown[] = [];
  await page.route("**/api/shops/nearby", (route) => {
    expect(route.request().method()).toBe("POST");
    expect(new URL(route.request().url()).search).toBe("");
    bodies.push(route.request().postDataJSON());
    return route.fulfill({ json: nearby() });
  });
  const response = await page.goto("/shops");
  expect(response?.headers()["permissions-policy"]).toContain(
    "geolocation=(self)",
  );
  await expect(
    page.getByRole("heading", { name: "Where are we digging?" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () =>
        (window as typeof window & { locationRequests: number })
          .locationRequests,
    ),
  ).toBe(0);
  expect(bodies).toHaveLength(0);
  await page.getByRole("button", { name: "Use my current location" }).click();
  await expect(
    page.getByRole("heading", { name: "3 places to dig" }),
  ).toBeVisible();
  expect(bodies).toEqual([
    { latitude: 40.71, longitude: -74.01, radiusKm: 10 },
  ]);
  expect(
    await page.evaluate(
      () =>
        (window as typeof window & { locationRequests: number })
          .locationRequests,
    ),
  ).toBe(1);
  await expect(page.locator(".shop-card h3")).toHaveText([
    "The Soul Crate",
    "Needle & Groove",
    "Long Play Records",
  ]);
  const directions = page
    .locator(".shop-card")
    .first()
    .getByRole("link", { name: "Directions" });
  const destination = new URL((await directions.getAttribute("href"))!);
  expect(destination.origin).toBe("https://www.google.com");
  expect(destination.pathname).toBe("/maps/dir/");
  expect(destination.searchParams.get("destination")).toBe(
    "40.71642,-74.00218",
  );
  expect(destination.searchParams.has("origin")).toBe(false);
  await expect(directions).toHaveAttribute("rel", "noopener noreferrer");
  await expect(
    page.getByText("Address details unavailable.", { exact: false }),
  ).toBeVisible();
  await expectNoOverflow(page);
  if (test.info().project.name === "desktop-chromium") {
    await page.screenshot({
      path: ".wax-local/shops-desktop.png",
      fullPage: true,
    });
  } else if (test.info().project.name === "mobile-webkit") {
    await page.screenshot({
      path: ".wax-local/shops-mobile.png",
      fullPage: true,
    });
  }
});

test("location denial offers manual area selection and provider retry keeps the query and shops", async ({
  page,
}) => {
  await mockLibrary(page);
  await page.addInitScript(() => {
    navigator.geolocation.getCurrentPosition = (_success, error) => {
      error?.({
        code: 1,
        message: "Permission denied",
        PERMISSION_DENIED: 1,
        POSITION_UNAVAILABLE: 2,
        TIMEOUT: 3,
      });
    };
  });
  const queries: string[] = [];
  await page.route("**/api/shops/places", (route) => {
    expect(route.request().method()).toBe("POST");
    expect(new URL(route.request().url()).search).toBe("");
    const { query } = route.request().postDataJSON();
    queries.push(query);
    return route.fulfill({
      json: { places: query === "Nowhereville" ? [] : places },
    });
  });
  let nearbyCalls = 0;
  await page.route("**/api/shops/nearby", (route) => {
    nearbyCalls++;
    const input = route.request().postDataJSON();
    expect(input.latitude).toBe(40.68);
    expect(input.longitude).toBe(-73.94);
    return nearbyCalls === 2
      ? route.fulfill({
          status: 503,
          json: {
            error:
              "Shop listings are temporarily unavailable. Please try again.",
          },
        })
      : route.fulfill({ json: nearby(input.radiusKm) });
  });
  await page.goto("/shops");
  const query = page.getByLabel("Or enter a city or postal code");
  const submit = page.getByRole("button", { name: "Find area" });
  await expect(submit).toBeDisabled();
  await query.fill("   ");
  await expect(submit).toBeDisabled();
  expect(queries).toHaveLength(0);
  await page.getByRole("button", { name: "Use my current location" }).click();
  await expect(page.locator(".shops-error").getByRole("alert")).toContainText(
    "Location access is off",
  );
  await query.fill("Nowhereville");
  await submit.click();
  await expect(
    page.getByText("No area found for", { exact: false }),
  ).toBeVisible();
  await expect(query).toHaveValue("Nowhereville");
  await query.fill("Brooklyn");
  await submit.click();
  await expect(
    page.getByRole("heading", { name: "Choose your area" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: places[1].label, exact: true }),
  ).toBeVisible();
  expect(nearbyCalls).toBe(0);
  await page
    .getByRole("button", { name: places[0].label, exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "3 places to dig" }),
  ).toBeVisible();
  await page.getByRole("radio", { name: "15 miles", exact: true }).check();
  await expect(page.locator(".shops-error").getByRole("alert")).toContainText(
    "Shop listings are temporarily unavailable",
  );
  await expect(query).toHaveValue("Brooklyn");
  await expect(page.locator(".shop-card h3")).toHaveText([
    "The Soul Crate",
    "Needle & Groove",
    "Long Play Records",
  ]);
  await expect(
    page.getByRole("link", { name: "Keep digging on Google Maps" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.locator(".shops-error").getByRole("alert")).toHaveCount(0);
  await expect(
    page.getByText("NEAREST FIRST · 15 MILES RADIUS, APPROX."),
  ).toBeVisible();
  expect(nearbyCalls).toBe(3);
  await expect(query).toHaveValue("Brooklyn");
  await expectNoOverflow(page);
});

test("a late location callback cannot overwrite a newer manual area", async ({
  page,
}) => {
  await mockLibrary(page);
  await page.addInitScript(() => {
    const state = window as typeof window & {
      finishLocation?: PositionCallback;
    };
    navigator.geolocation.getCurrentPosition = (success) => {
      state.finishLocation = success;
    };
  });
  await page.route("**/api/shops/places", (route) =>
    route.fulfill({ json: { places } }),
  );
  const bodies: unknown[] = [];
  await page.route("**/api/shops/nearby", (route) => {
    bodies.push(route.request().postDataJSON());
    return route.fulfill({ json: { ...nearby(), shops: [] } });
  });
  await page.goto("/shops");
  await page.getByRole("button", { name: "Use my current location" }).click();
  await expect(
    page.getByRole("button", { name: "Finding your area…" }),
  ).toBeVisible();
  await page.getByLabel("Or enter a city or postal code").fill("Brooklyn");
  await page.getByRole("button", { name: "Find area" }).click();
  await page
    .getByRole("button", { name: places[0].label, exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "No mapped record shops in this area yet.",
    }),
  ).toBeVisible();
  await page.evaluate(() => {
    (
      window as typeof window & { finishLocation?: PositionCallback }
    ).finishLocation?.({
      coords: {
        latitude: 51.507351,
        longitude: -0.127758,
        accuracy: 1,
        altitude: null,
        altitudeAccuracy: null,
        heading: null,
        speed: null,
        toJSON: () => ({}),
      },
      timestamp: Date.now(),
      toJSON: () => ({}),
    });
  });
  await expect(page.locator(".shops-results-heading")).toContainText(
    places[0].label,
  );
  expect(bodies).toEqual([
    { latitude: 40.68, longitude: -73.94, radiusKm: 10 },
  ]);
  await expect(
    page.getByRole("link", { name: "Look on Google Maps" }),
  ).toBeVisible();
});
