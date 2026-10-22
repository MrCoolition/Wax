import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";
import {
  placesSchema,
  readShopRequest,
  searchShopPlaces,
} from "@/server/shops";

export const runtime = "nodejs";
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const input = placesSchema.parse(await readShopRequest(request));
    await consumeIntegrationLimit(collector.id, "shop-places", 20, 60);
    return { places: await searchShopPlaces(input) };
  });
}
