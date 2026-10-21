import { limitRegistration } from "@/server/invitations";
import { handleApi } from "@/server/http";
import { resetPassword } from "@/server/recovery";

export async function POST(request: Request) {
  return handleApi(request, async () => {
    await limitRegistration(request);
    return resetPassword(await request.json());
  });
}
