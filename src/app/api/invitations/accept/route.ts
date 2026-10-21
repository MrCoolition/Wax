import {
  acceptInvitation,
  inspectInvitation,
  limitRegistration,
} from "@/server/invitations";
import { handleApi } from "@/server/http";
export async function GET(request: Request) {
  return handleApi(request, async () => {
    await limitRegistration(request);
    return inspectInvitation(
      new URL(request.url).searchParams.get("token") ?? "",
    );
  });
}
export async function POST(request: Request) {
  return handleApi(request, async () => {
    await limitRegistration(request);
    return acceptInvitation(await request.json());
  });
}
