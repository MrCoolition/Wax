/** Public catalog evidence obtained during this Wizard answer. */
export interface WizardCatalogSource {
  id: string;
  kind: "master" | "release" | "artist";
  title: string;
  url: string;
}
