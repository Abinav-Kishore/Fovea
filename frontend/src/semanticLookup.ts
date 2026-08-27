/**
 * Class-id -> class-name lookup, populated from GET /api/semantic-classes
 * (which itself reuses src/semantic_labels.LABEL_DICT). We deliberately do
 * NOT hardcode SemanticPOSS's id->name table here -- the backend is the only
 * source of truth for it.
 */
import type { SemanticClass } from "./api/types";

const lookup = new Map<number, string>();

export function initSemanticLookup(classes: SemanticClass[]) {
  lookup.clear();
  for (const c of classes) lookup.set(c.class_id, c.class_name);
}

export function classNameForId(id: number): string {
  return lookup.get(id) ?? "unknown";
}
