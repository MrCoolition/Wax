import { requireCollector } from "@/server/auth";
import { updateCrate, deleteCrate } from "@/server/collection";
import { handleApi } from "@/server/http";
type Context = { params: Promise<{ id: string }> };
export async function PATCH(request: Request, context: Context) {
  return handleApi(request, async () =>
    updateCrate(
      (await requireCollector()).id,
      (await context.params).id,
      await request.json(),
    ),
  );
}
export async function DELETE(request: Request, context: Context) {
  return handleApi(request, async () =>
    deleteCrate((await requireCollector()).id, (await context.params).id),
  );
}
