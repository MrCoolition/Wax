export type VerifiedRecordShop = {
  slug: string;
  name: string;
  latitude: number;
  longitude: number;
  address: string;
  website: string;
  sourceUrl: string;
  verifiedOn: string;
  coordinateSourceUrl: string;
  coordinateAccuracy: "address-interpolated" | "mapped-location";
  osmId?: `node/${number}` | `way/${number}` | `relation/${number}`;
};

// Supplemental listings for gaps in the community map, not a complete directory.
// Recheck the seller's inventory AND address before renewing verifiedOn. The
// caller expires old entries. Census points are interpolated street addresses,
// not surveyed entrances; distances and directions remain approximate.
// At admission, check the map for an existing location and set osmId when one
// exists. Otherwise the directory slug is its stable identity, even if the map
// later discovers it; never silently change that identity and split saved notes.
const censusAddress = (address: string) =>
  `https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?${new URLSearchParams({ address, benchmark: "Public_AR_Current", format: "json" })}`;

export const VERIFIED_RECORD_SHOPS: readonly VerifiedRecordShop[] = [
  {
    slug: "spoonful-records-columbus-oh",
    name: "Spoonful Records",
    latitude: 39.960670930193,
    longitude: -82.996200123499,
    address: "144 E State Street, Columbus, Ohio 43215",
    website: "https://spoonfulrecordsohio.com/",
    // The store's own page states its vinyl inventory and current address.
    sourceUrl: "https://spoonfulrecordsohio.com/",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl: censusAddress(
      "144 E State Street, Columbus, OH 43215",
    ),
    coordinateAccuracy: "address-interpolated",
  },
  {
    slug: "lovelight-records-and-art-columbus-oh",
    name: "Lovelight Records and Art",
    latitude: 40.057465189411,
    longitude: -83.000528962819,
    address: "4540 Indianola Avenue, Columbus, Ohio 43214",
    website: "https://lovelightrecordsandart.com/",
    // Public profile: records, art and culture; same address as its own website.
    sourceUrl: "https://www.instagram.com/lovelightrecordsandart/",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl: censusAddress(
      "4540 Indianola Avenue, Columbus, OH 43214",
    ),
    coordinateAccuracy: "address-interpolated",
  },
  {
    slug: "needle-exchange-records-and-tapes-columbus-oh",
    name: "The Needle Exchange Records & Tapes",
    latitude: 40.051876672553,
    longitude: -83.000028330849,
    address: "4290 Indianola Avenue, Suite 201, Columbus, Ohio 43214",
    website: "https://www.instagram.com/needleexchangerecords/",
    // The shop's public profile lists records and tapes at 4290 Indianola,
    // upstairs; suite number is corroborated by Women in Vinyl's store guide.
    sourceUrl: "https://www.instagram.com/needleexchangerecords/",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl: censusAddress(
      "4290 Indianola Avenue, Columbus, OH 43214",
    ),
    coordinateAccuracy: "address-interpolated",
  },
  {
    slug: "pats-endangered-species-delaware-oh",
    name: "Pat's Endangered Species",
    latitude: 40.300145796508,
    longitude: -83.068277716613,
    address: "11 W Winter Street, Delaware, Ohio 43015",
    website: "https://www.instagram.com/patsendangeredspecies_/",
    // Main Street Delaware confirms the address and records. The shop's own
    // profile also advertises new and used CDs, records and cassettes.
    sourceUrl:
      "https://www.mainstreetdelaware.com/places/united-states/ohio/delaware/shopping/pats-endangered-species-the-last-record-store-on-earth/",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl: censusAddress(
      "11 W Winter Street, Delaware, OH 43015",
    ),
    coordinateAccuracy: "address-interpolated",
  },
  {
    slug: "legend-records-mount-gilead-oh",
    name: "Legend Records",
    latitude: 40.547934595259,
    longitude: -82.8288792105,
    address: "37 S Main Street, Mount Gilead, Ohio 43338",
    website: "https://www.instagram.com/legendrecordsohio/",
    // First-person interview with the owner confirms physical-shop vinyl
    // inventory. Address is also listed in Women in Vinyl's Columbus guide.
    sourceUrl: "https://vinylpackman.com/store/1944",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl: censusAddress("37 S Main St, Mount Gilead, OH 43338"),
    coordinateAccuracy: "address-interpolated",
  },
  {
    slug: "toxic-beauty-records-yellow-springs-oh",
    name: "Toxic Beauty Records",
    latitude: 39.8060797,
    longitude: -83.888874,
    address: "220 Xenia Avenue, Yellow Springs, Ohio 45387",
    website: "https://vinylgram.com/",
    // County tourism listing confirms new/used records and the shop address;
    // the linked shop storefront also lists current vinyl releases.
    sourceUrl:
      "https://www.greenecountyohio.org/listing/toxic-beauty-records/1095/",
    verifiedOn: "2026-10-23",
    coordinateSourceUrl:
      "https://photon.komoot.io/reverse?lat=39.96&lon=-83&radius=100&limit=100&include=osm.shop.music,osm.shop.musical_instrument&lang=en",
    coordinateAccuracy: "mapped-location",
    osmId: "way/680555954",
  },
];
