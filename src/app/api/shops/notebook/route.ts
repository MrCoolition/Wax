import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";
import {
  deleteShopNotebook,
  deleteShopSchema,
  getShopNotebook,
  readNotebookRequest,
  saveShopNotebook,
  saveShopSchema,
} from "@/server/shop-notebook";

export const runtime = "nodejs";

export async function GET(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    await consumeIntegrationLimit(collector.id, "shops-notebook-read", 120, 60);
    return getShopNotebook(collector.id);
  });
}

export async function PUT(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const input = saveShopSchema.parse(await readNotebookRequest(request));
    await consumeIntegrationLimit(collector.id, "shops-notebook-write", 60, 60);
    return saveShopNotebook(collector.id, input);
  });
}

export async function DELETE(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const input = deleteShopSchema.parse(await readNotebookRequest(request));
    await consumeIntegrationLimit(collector.id, "shops-notebook-write", 60, 60);
    return deleteShopNotebook(collector.id, input);
  });
}
