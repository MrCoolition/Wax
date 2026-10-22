import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";
import { findNearbyShops, nearbySchema, readShopRequest } from "@/server/shops";

export const runtime = "nodejs";
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const input = nearbySchema.parse(await readShopRequest(request));
    await consumeIntegrationLimit(collector.id, "shops-nearby", 20, 60);
    return findNearbyShops(input);
  });
}
