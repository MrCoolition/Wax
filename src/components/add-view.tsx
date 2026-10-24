"use client";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  Check,
  Disc3,
  ImagePlus,
  PencilLine,
  ScanLine,
  Search,
  X,
} from "lucide-react";
import type {
  CatalogCandidate,
  CopyFields,
  PhotoIdentification,
  RecordCopy,
} from "@/lib/contracts";
import type { AlbumSearchResult, CatalogAlbum } from "@/lib/catalog-search";
import { DEFAULT_COPY, EMPTY_CANDIDATE } from "@/lib/contracts";
import { api, ApiError, useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  ErrorNotice,
  Field,
  PageHeader,
  SourceCredit,
} from "./ui";
import { CandidateFields, CopyForm } from "./record-form";
import {
  CatalogFinder,
  CatalogFinderDialog,
  ManualCatalogSuggestions,
  type CatalogSearchDraft,
} from "./catalog-finder";
import styles from "./catalog-finder.module.css";

type Mode = "search" | "scan" | "photo" | "manual";
async function compressImage(file: File): Promise<File> {
  if (!file.type.startsWith("image/"))
    throw new Error("Choose a photo in JPEG, PNG, or WebP format.");
  const bitmap = await createImageBitmap(file).catch(() => {
    throw new Error(
      "That image format could not be opened. Try a JPEG or PNG photo.",
    );
  });
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1800 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (value) =>
        value
          ? resolve(value)
          : reject(new Error("Could not prepare that photo.")),
      "image/jpeg",
      0.88,
    ),
  );
  if (blob.size > 3 * 1024 * 1024)
    throw new Error("Please choose a smaller photo, under 3 MB.");
  return new File([blob], "record-photo.jpg", { type: "image/jpeg" });
}
function BarcodeCamera({
  onResult,
  onClose,
}: {
  onResult: (value: string) => void;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const resultHandler = useRef(onResult);
  resultHandler.current = onResult;
  const [error, setError] = useState("");
  useEffect(() => {
    let stopped = false;
    let controls: { stop: () => void } | undefined;
    const currentVideo = video.current;
    void import("@zxing/browser")
      .then(async ({ BrowserMultiFormatOneDReader }) => {
        if (stopped || !currentVideo) return;
        const reader = new BrowserMultiFormatOneDReader();
        const active = await reader.decodeFromConstraints(
          { audio: false, video: { facingMode: { ideal: "environment" } } },
          currentVideo,
          (result, _, scanner) => {
            if (result && !stopped) {
              stopped = true;
              scanner.stop();
              resultHandler.current(result.getText());
            }
          },
        );
        controls = active;
        if (stopped) active.stop();
      })
      .catch((err) => {
        if (!stopped)
          setError(
            err?.name === "NotAllowedError"
              ? "Camera access was denied. Allow camera access in your browser, or enter the barcode below."
              : "We couldn’t start that camera. You can still type the barcode below.",
          );
      });
    return () => {
      stopped = true;
      controls?.stop();
      const stream = currentVideo?.srcObject;
      if (typeof MediaStream !== "undefined" && stream instanceof MediaStream)
        stream.getTracks().forEach((track) => track.stop());
    };
  }, []);
  return (
    <div className="camera-box">
      <div className="camera-top">
        <span>
          <ScanLine size={17} /> FIND THE BARCODE
        </span>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Stop camera"
        >
          <X size={20} />
        </button>
      </div>
      <video ref={video} muted playsInline autoPlay />
      <div className="scan-frame" aria-hidden="true" />
      <p>Keep the barcode flat and steady in the frame.</p>
      <ErrorNotice>{error}</ErrorNotice>
    </div>
  );
}
export function AddView() {
  const params = useSearchParams();
  const router = useRouter();
  const { refresh, data } = useLibrary();
  const initialMode = params.get("mode");
  const [mode, setMode] = useState<Mode>(
    ["scan", "photo", "manual"].includes(initialMode || "")
      ? (initialMode as Mode)
      : "search",
  );
  const [barcode, setBarcode] = useState("");
  const [scanSearch, setScanSearch] = useState<{
    value: string;
    key: number;
  } | null>(null);
  const [camera, setCamera] = useState(false);
  const [candidate, setCandidate] = useState<CatalogCandidate | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const [manualDraft, setManualDraft] = useState<CatalogCandidate>({
    ...EMPTY_CANDIDATE,
  });
  const [copy, setCopy] = useState<CopyFields>({ ...DEFAULT_COPY });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState("");
  const [photoResult, setPhotoResult] = useState<
    (PhotoIdentification & { revision: number }) | null
  >(null);
  const [allowDuplicate, setAllowDuplicate] = useState(false);
  const [duplicate, setDuplicate] = useState(false);
  const [crateIds, setCrateIds] = useState<string[]>([]);
  const [lookup, setLookup] = useState<{
    search: CatalogSearchDraft;
    albums?: AlbumSearchResult;
    album?: CatalogAlbum;
  } | null>(null);
  const requestKey = useRef<string | null>(null);
  const photoRequest = useRef<{
    sequence: number;
    controller: AbortController | null;
  }>({ sequence: 0, controller: null });
  const reviewHeading = useRef<HTMLHeadingElement>(null);
  const wizardArtist = params.get("artist")?.slice(0, 200) || "";
  const wizardTitle = params.get("title")?.slice(0, 200) || "";
  const wizardMaster = Number(params.get("master"));
  const wizardSearch: CatalogSearchDraft =
    wizardArtist || wizardTitle
      ? { mode: "details", artist: wizardArtist, title: wizardTitle }
      : { mode: "album" };
  const wizardAlbum: CatalogAlbum | undefined =
    Number.isSafeInteger(wizardMaster) && wizardMaster > 0
      ? {
          masterId: wizardMaster,
          artist: wizardArtist || "Catalog artist",
          title: wizardTitle || "Selected album",
          originalYear: null,
          coverUrl: null,
          sourceUrl: `https://www.discogs.com/master/${wizardMaster}`,
        }
      : undefined;

  useEffect(() => {
    if (!photo) {
      setPhotoPreview("");
      return;
    }
    const url = URL.createObjectURL(photo);
    setPhotoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);
  useEffect(
    () => () => {
      photoRequest.current.sequence += 1;
      photoRequest.current.controller?.abort();
    },
    [],
  );
  function cancelPhoto() {
    photoRequest.current.sequence += 1;
    photoRequest.current.controller?.abort();
    setBusy("");
  }
  function startReview(value: CatalogCandidate) {
    cancelPhoto();
    if (candidate && !candidate.discogsId) setManualDraft(candidate);
    setCandidate(value);
    setCopy((current) => ({ ...current, pressingConfirmed: false }));
    setReviewing(true);
    setLookup(null);
    setCamera(false);
    setDuplicate(false);
    setAllowDuplicate(false);
    setError("");
    requestKey.current = null;
    requestAnimationFrame(() => {
      reviewHeading.current?.focus({ preventScroll: true });
      reviewHeading.current?.scrollIntoView({ block: "start" });
    });
  }
  function enterManual(value = manualDraft) {
    cancelPhoto();
    setCandidate(value);
    setManualDraft(value);
    setCopy((current) => ({ ...current, pressingConfirmed: false }));
    setReviewing(true);
    setCamera(false);
    setDuplicate(false);
    setAllowDuplicate(false);
    setError("");
    requestKey.current = null;
  }
  function openLookup(albums?: AlbumSearchResult, album?: CatalogAlbum) {
    if (!candidate) return;
    const search: CatalogSearchDraft =
      candidate.catalogNumber.trim() && !album
        ? {
            mode: "catalogNumber",
            catalogNumber: candidate.catalogNumber,
            artist: candidate.artist,
            title: candidate.title,
          }
        : {
            mode: "details",
            artist: candidate.artist,
            title: candidate.title,
            catalogNumber: candidate.catalogNumber,
          };
    setLookup({
      search,
      albums: search.mode === "catalogNumber" ? undefined : albums,
      album,
    });
  }
  async function identify() {
    if (!photo) return;
    photoRequest.current.controller?.abort();
    const controller = new AbortController();
    const sequence = ++photoRequest.current.sequence;
    photoRequest.current.controller = controller;
    const timeout = window.setTimeout(() => controller.abort("timeout"), 60000);
    setBusy("photo");
    setError("");
    try {
      let currentPhotoId = photoId;
      if (!currentPhotoId) {
        const prepared = await compressImage(photo);
        if (controller.signal.aborted) return;
        const form = new FormData();
        form.append("file", prepared);
        const uploaded = await api<{ id: string; url: string }>("/api/photos", {
          method: "POST",
          body: form,
          signal: controller.signal,
        });
        if (sequence !== photoRequest.current.sequence) return;
        currentPhotoId = uploaded.id;
        setPhotoId(currentPhotoId);
      }
      const result = await api<PhotoIdentification>("/api/identify", {
        method: "POST",
        body: JSON.stringify({ photoId: currentPhotoId }),
        signal: controller.signal,
      });
      if (sequence !== photoRequest.current.sequence) return;
      setPhotoResult({ ...result, revision: sequence });
    } catch (err) {
      if (
        sequence !== photoRequest.current.sequence ||
        (controller.signal.aborted && controller.signal.reason !== "timeout")
      )
        return;
      setError(
        (controller.signal.reason === "timeout"
          ? "Recognition took too long."
          : err instanceof Error
            ? err.message
            : "Could not identify that photo.") +
          " Your photo is still here. Retry, or choose Add by hand to complete the details.",
      );
    } finally {
      window.clearTimeout(timeout);
      if (sequence === photoRequest.current.sequence) setBusy("");
    }
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!candidate || busy === "save") return;
    setBusy("save");
    setError("");
    requestKey.current ||= crypto.randomUUID();
    try {
      let attachedPhotoId = photoId;
      if (photo && !attachedPhotoId) {
        const prepared = await compressImage(photo);
        const form = new FormData();
        form.append("file", prepared);
        const uploaded = await api<{ id: string }>("/api/photos", {
          method: "POST",
          body: form,
        });
        attachedPhotoId = uploaded.id;
        setPhotoId(attachedPhotoId);
      }
      const result = await api<RecordCopy>("/api/records", {
        method: "POST",
        body: JSON.stringify({
          idempotencyKey: requestKey.current,
          candidate: { ...candidate, genres: candidate.genres.filter(Boolean) },
          copy,
          allowDuplicate,
          photoIds: attachedPhotoId ? [attachedPhotoId] : [],
          crateIds,
        }),
      });
      await refresh();
      router.push("/records/" + result.id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setDuplicate(true);
        requestKey.current = null;
      }
      setError(
        err instanceof Error ? err.message : "Could not save your record.",
      );
    } finally {
      setBusy("");
    }
  }
  const existing =
    candidate &&
    data?.records.some((record) =>
      candidate.discogsId
        ? record.discogsId === candidate.discogsId
        : record.title.toLowerCase() === candidate.title.toLowerCase() &&
          record.artist.toLowerCase() === candidate.artist.toLowerCase(),
    );
  const photoDraft: CatalogCandidate = {
    ...EMPTY_CANDIDATE,
    artist: photoResult?.clues.artist || "",
    title: photoResult?.clues.title || "",
    label: photoResult?.clues.label || "",
    catalogNumber: photoResult?.clues.catalogNumber || "",
  };
  const photoSearch: CatalogSearchDraft = photoResult?.clues.catalogNumber
    ? {
        mode: "catalogNumber",
        catalogNumber: photoResult.clues.catalogNumber,
        artist: photoResult.clues.artist || "",
        title: photoResult.clues.title || "",
      }
    : photoResult?.clues.barcode
      ? {
          mode: "barcode",
          barcode: photoResult.clues.barcode,
          artist: photoResult.clues.artist || "",
          title: photoResult.clues.title || "",
        }
      : {
          mode: "details",
          artist: photoResult?.clues.artist || "",
          title: photoResult?.clues.title || "",
        };
  return (
    <>
      <PageHeader
        eyebrow="NEW ARRIVALS / MAKE ROOM ON THE SHELF"
        title="Add some wax"
        description="Find the album. Check the pressing. Make it yours."
      />
      <div className="flow-steps">
        <span className={!reviewing ? "current" : "complete"}>
          <b>{reviewing ? <Check size={13} /> : "1"}</b> Find your record
        </span>
        <i />
        <span className={reviewing ? "current" : ""}>
          <b>2</b> The pressing & your copy
        </span>
        <i />
        <span>
          <b>3</b> Into the collection
        </span>
      </div>
      {reviewing && candidate && (
        <form onSubmit={save} className="review-layout">
          <div className="review-main">
            <button
              type="button"
              className="text-button"
              disabled={busy === "save"}
              onClick={() => {
                setReviewing(false);
                setError("");
              }}
            >
              <ArrowLeft size={15} /> Back to finding your record
            </button>
            <div className="section-title">
              <span>01 / THE RELEASE</span>
              <h2 ref={reviewHeading} tabIndex={-1}>
                Let’s get the details right.
              </h2>
            </div>
            {candidate.discogsId ? (
              <div className={styles.reviewLookup}>
                <p>
                  Catalog details are ready for your review. Check the label,
                  country, and catalog number against your copy.
                </p>
                <Button
                  type="button"
                  variant="secondary"
                  disabled={busy === "save"}
                  onClick={() => openLookup()}
                >
                  Find another pressing
                </Button>
              </div>
            ) : (
              <ManualCatalogSuggestions
                artist={candidate.artist}
                title={candidate.title}
                onBrowse={(albums) => openLookup(albums)}
                onAlbum={(album, albums) => openLookup(albums, album)}
              />
            )}
            <CandidateFields
              value={candidate}
              onChange={(value) => {
                setCandidate(value);
                if (!value.discogsId) setManualDraft(value);
              }}
            />
            {candidate.discogsId &&
              (manualDraft.artist || manualDraft.title) && (
                <div className={styles.restore}>
                  <button
                    type="button"
                    className="text-button"
                    disabled={busy === "save"}
                    onClick={() => enterManual(manualDraft)}
                  >
                    <ArrowLeft size={14} /> Restore manual details
                  </button>
                  <p className="small-note">
                    Your condition, notes, photos, and crate choices stay with
                    this copy.
                  </p>
                </div>
              )}
            <div className="section-title">
              <span>02 / YOUR COPY</span>
              <h2>Same record. Your story.</h2>
            </div>
            <CopyForm value={copy} onChange={setCopy} />
            {Boolean(data?.crates.length) && (
              <fieldset className="crate-checkboxes">
                <legend>File it in a crate</legend>
                {data!.crates.map((crate) => (
                  <label key={crate.id}>
                    <input
                      type="checkbox"
                      checked={crateIds.includes(crate.id)}
                      onChange={(event) =>
                        setCrateIds(
                          event.target.checked
                            ? [...crateIds, crate.id]
                            : crateIds.filter((id) => id !== crate.id),
                        )
                      }
                    />
                    <i style={{ background: crate.color }} />
                    {crate.name}
                  </label>
                ))}
              </fieldset>
            )}
            {(existing || duplicate) && (
              <label className="checkbox-row duplicate-box">
                <input
                  type="checkbox"
                  checked={allowDuplicate}
                  onChange={(event) => setAllowDuplicate(event.target.checked)}
                />
                <span>
                  <strong>Looks like this one’s already on your shelf.</strong>
                  <small>
                    Check this to add another physical copy. Your existing copy
                    stays as it is.
                  </small>
                </span>
              </label>
            )}
            <ErrorNotice>{error}</ErrorNotice>
            <div className="form-actions">
              <Button
                type="submit"
                busy={busy === "save"}
                disabled={Boolean((existing || duplicate) && !allowDuplicate)}
              >
                Add to my collection <ArrowRight size={17} />
              </Button>
              <span>Saved privately. Always yours.</span>
            </div>
          </div>
          <aside className="review-sleeve">
            <Cover
              record={{
                ...candidate,
                coverUrl: photoPreview || candidate.coverUrl,
              }}
            />
            <span className="eyebrow">YOUR NEXT GOOD FIND</span>
            <h3>{candidate.title || "Untitled record"}</h3>
            <p>{candidate.artist || "Unknown artist"}</p>
            <div className="pressing-status">
              <span
                className={
                  "status-dot " + (copy.pressingConfirmed ? "green" : "")
                }
              />
              {copy.pressingConfirmed
                ? "Pressing checked"
                : "Pressing unconfirmed"}
            </div>
            <SourceCredit sourceUrl={candidate.sourceUrl} />
            <p className="small-note">
              A cover can tell us the album. The label, catalog number, and
              runout tell us the pressing.
            </p>
          </aside>
        </form>
      )}
      <div hidden={reviewing}>
        {candidate && (
          <div className={styles.resume}>
            <span>
              Your draft of <strong>{candidate.title || "this record"}</strong>{" "}
              is still here.
            </span>
            <button
              type="button"
              className="text-button"
              onClick={() => setReviewing(true)}
            >
              Return to your draft <ArrowRight size={14} />
            </button>
          </div>
        )}
        <div className="capture-tabs">
          {[
            { id: "search", label: "Search the catalog", icon: Search },
            { id: "scan", label: "Scan a barcode", icon: ScanLine },
            { id: "photo", label: "Snap a photo", icon: Camera },
            { id: "manual", label: "Add by hand", icon: PencilLine },
          ].map(({ id, label, icon: Icon }) => (
            <button
              type="button"
              key={id}
              className={mode === id ? "active" : ""}
              aria-pressed={mode === id}
              onClick={() => {
                cancelPhoto();
                setMode(id as Mode);
                setCamera(false);
                setError("");
              }}
            >
              <Icon size={20} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <div hidden={mode !== "search"}>
          <div className="section-title">
            <span>THE DIG STARTS HERE</span>
            <h2>What are we looking for?</h2>
          </div>
          <CatalogFinder
            active={!reviewing && mode === "search"}
            key={params.toString()}
            initialSearch={wizardSearch}
            initialAlbum={wizardAlbum}
            autoSearch={Boolean(wizardArtist || wizardTitle)}
            onSelect={startReview}
            onManual={() => enterManual()}
          />
        </div>
        <div hidden={mode !== "scan"}>
          <section className="capture-panel">
            <span className="eyebrow">A QUICK FLIP TO THE BACK</span>
            <h2>Scan it. Then check it.</h2>
            <p>
              A barcode can match several pressings. You’ll choose the right one
              next.
            </p>
            {camera && !reviewing ? (
              <BarcodeCamera
                onClose={() => setCamera(false)}
                onResult={(value) => {
                  setBarcode(value);
                  setCamera(false);
                  setScanSearch({ value, key: Date.now() });
                }}
              />
            ) : (
              <Button type="button" onClick={() => setCamera(true)}>
                <Camera size={18} /> Open camera
              </Button>
            )}
            <form
              className="barcode-manual"
              onSubmit={(event) => {
                event.preventDefault();
                setScanSearch({ value: barcode, key: Date.now() });
              }}
            >
              <Field label="Or type the barcode">
                <input
                  value={barcode}
                  inputMode="numeric"
                  required
                  maxLength={40}
                  onChange={(event) => setBarcode(event.target.value)}
                  placeholder="Keep any leading zeros"
                />
              </Field>
              <Button variant="secondary" type="submit">
                Find matches <ArrowRight size={16} />
              </Button>
            </form>
          </section>
          {scanSearch && (
            <CatalogFinder
              active={!reviewing && mode === "scan"}
              key={scanSearch.key}
              initialSearch={{ mode: "barcode", barcode: scanSearch.value }}
              autoSearch
              onSelect={startReview}
              onManual={() => enterManual()}
            />
          )}
        </div>
        <div hidden={mode !== "photo"}>
          <section className="capture-panel">
            <span className="eyebrow">LET THE SLEEVE DO THE TALKING</span>
            <h2>Recognize a record.</h2>
            <p>
              Photograph the cover or label in good light. Correct the clues
              below, compare the pressings, and choose what matches your copy.
            </p>
            <div className="photo-capture">
              <label
                className={"photo-drop " + (photoPreview ? "has-photo" : "")}
              >
                {photoPreview ? (
                  <img src={photoPreview} alt="Your selected record photo" />
                ) : (
                  <>
                    <ImagePlus size={35} />
                    <strong>Choose a photo or open your camera</strong>
                    <span>JPEG, PNG, or WebP · we’ll size it for you</span>
                  </>
                )}
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  capture="environment"
                  onChange={(event) => {
                    cancelPhoto();
                    setPhoto(event.target.files?.[0] || null);
                    setPhotoId(null);
                    setPhotoResult(null);
                    setError("");
                  }}
                />
              </label>
              {photo && (
                <Button
                  type="button"
                  onClick={() => void identify()}
                  busy={busy === "photo"}
                >
                  <Camera size={17} />
                  {busy === "photo"
                    ? "Looking closely…"
                    : "Identify this record"}
                </Button>
              )}
            </div>
            <ErrorNotice>{error}</ErrorNotice>
          </section>
          {photoResult && (
            <>
              <div className="notice">
                <strong>Photo clues, ready to refine.</strong>
                <p>
                  {[
                    photoResult.clues.artist,
                    photoResult.clues.title,
                    photoResult.clues.uncertainty,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
                <p>
                  These are suggestions. Edit the search clues or narrow by
                  country, label, and year below.
                </p>
              </div>
              <CatalogFinder
                active={!reviewing && mode === "photo"}
                key={photoResult.revision}
                initialSearch={photoSearch}
                initialCandidates={photoResult.candidates}
                onSelect={startReview}
                onManual={() => enterManual(photoDraft)}
              />
            </>
          )}
        </div>
        <div hidden={mode !== "manual"}>
          <section className="capture-panel">
            <span className="eyebrow">YOU KNOW YOUR RECORDS</span>
            <h2>Keep it simple.</h2>
            <p>
              Start with the artist and title. We’ll look for catalog matches
              while you fill in your copy. Choose a match when it’s right, or
              keep your own details.
            </p>
            <Button type="button" onClick={() => enterManual()}>
              Enter the details <PencilLine size={17} />
            </Button>
          </section>
        </div>
        <div className="capture-footnote">
          <Disc3 size={20} />
          <p>
            Different pressing? Different story. We’ll keep your physical copies
            separate, even when the album is the same.
          </p>
        </div>
      </div>
      {lookup && (
        <CatalogFinderDialog
          open
          onOpenChange={(open) => {
            if (!open) setLookup(null);
          }}
          initialSearch={lookup.search}
          initialAlbums={lookup.albums}
          initialAlbum={lookup.album}
          autoSearch={!lookup.albums && !lookup.album}
          onSelect={startReview}
        />
      )}
    </>
  );
}
