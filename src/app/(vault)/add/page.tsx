import { Suspense } from "react";
import { AddView } from "@/components/add-view";
export default function Page() {
  return (
    <Suspense fallback={<p>Opening the crate…</p>}>
      <AddView />
    </Suspense>
  );
}
