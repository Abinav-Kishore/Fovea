/**
 * Client-side decoder for the FVCH1 binary chunk format produced by
 * src/spatial/chunks.py (`serialize_chunk`). Mirrors that module's format
 * EXACTLY -- see its docstring for the format rationale. This file makes no
 * FOVEA/hierarchy/LOD decision; it only decodes bytes the backend already
 * produced from an already-selected LOD frontier.
 *
 * Decodes into a STRUCT-OF-ARRAYS (typed arrays), never an array of per-chunk
 * JS objects -- tens of thousands of chunks as one JS object per chunk would
 * be exactly the "huge JSON payload" problem this format exists to avoid,
 * just moved client-side. `chunkInfoAt()` builds a single plain object only
 * when one chunk needs to be inspected (e.g. after a click).
 *
 * Wire layout (little-endian, matches src/spatial/chunks.py's `_MAGIC` /
 * `_CHUNK_STRUCT` exactly -- "version" is embedded IN the magic string
 * itself, "FVCH1" = format v1; there is no separate version field on the
 * wire, so none is invented here):
 *
 *   6 bytes  magic "FVCH1\0"
 *   uint32   chunk_count
 *   uint32   string_count
 *   string_count x (uint16 length + UTF-8 bytes)   <- interned string table
 *   chunk_count x 94-byte fixed record:
 *     uint8  level
 *     uint8  is_leaf
 *     int32  band_id
 *     int32  ix
 *     int32  iy
 *     float32 cell_size_m
 *     float32 x_min, x_max, y_min, y_max
 *     float32 center_x, center_y, range_m
 *     uint32 point_count
 *     float32 elevation_min, elevation_max, elevation_median
 *     float32 semantic_priority, majority_class_fraction
 *     float32 geometry_score, dynamic_score, information_score
 *     uint16 semantic_class_idx, terrain_class_idx, obstacle_class_idx,
 *            resolution_label_idx, label_source_idx, source_idx
 */

const MAGIC = [0x46, 0x56, 0x43, 0x48, 0x31, 0x00]; // "FVCH1\0"
const HEADER_SIZE = 6 + 4 + 4; // magic + chunk_count + string_count
const RECORD_SIZE = 94;

export class Fvch1DecodeError extends Error {
  constructor(message: string) {
    super(`FVCH1 payload rejected: ${message}`);
    this.name = "Fvch1DecodeError";
  }
}

/** Struct-of-arrays decode of one chunk set -- one typed-array slot per
 * field per chunk, indexed 0..count-1. String-valued fields are stored as
 * small integer indices into `stringTable` (see chunkInfoAt to resolve
 * one). */
export interface DecodedChunkSet {
  count: number;
  level: Uint8Array;
  isLeaf: Uint8Array;
  bandId: Int32Array;
  ix: Int32Array;
  iy: Int32Array;
  cellSizeM: Float32Array;
  xMin: Float32Array;
  xMax: Float32Array;
  yMin: Float32Array;
  yMax: Float32Array;
  centerX: Float32Array;
  centerY: Float32Array;
  rangeM: Float32Array;
  pointCount: Uint32Array;
  elevationMin: Float32Array;
  elevationMax: Float32Array;
  elevationMedian: Float32Array;
  semanticPriority: Float32Array;
  majorityClassFraction: Float32Array;
  geometryScore: Float32Array;
  dynamicScore: Float32Array;
  informationScore: Float32Array;
  semanticClassIdx: Uint16Array;
  terrainClassIdx: Uint16Array;
  obstacleClassIdx: Uint16Array;
  resolutionLabelIdx: Uint16Array;
  labelSourceIdx: Uint16Array;
  sourceIdx: Uint16Array;
  stringTable: string[];
}

function emptySet(): DecodedChunkSet {
  return {
    count: 0,
    level: new Uint8Array(0),
    isLeaf: new Uint8Array(0),
    bandId: new Int32Array(0),
    ix: new Int32Array(0),
    iy: new Int32Array(0),
    cellSizeM: new Float32Array(0),
    xMin: new Float32Array(0),
    xMax: new Float32Array(0),
    yMin: new Float32Array(0),
    yMax: new Float32Array(0),
    centerX: new Float32Array(0),
    centerY: new Float32Array(0),
    rangeM: new Float32Array(0),
    pointCount: new Uint32Array(0),
    elevationMin: new Float32Array(0),
    elevationMax: new Float32Array(0),
    elevationMedian: new Float32Array(0),
    semanticPriority: new Float32Array(0),
    majorityClassFraction: new Float32Array(0),
    geometryScore: new Float32Array(0),
    dynamicScore: new Float32Array(0),
    informationScore: new Float32Array(0),
    semanticClassIdx: new Uint16Array(0),
    terrainClassIdx: new Uint16Array(0),
    obstacleClassIdx: new Uint16Array(0),
    resolutionLabelIdx: new Uint16Array(0),
    labelSourceIdx: new Uint16Array(0),
    sourceIdx: new Uint16Array(0),
    stringTable: [],
  };
}

/** Decodes one FVCH1 payload. Throws `Fvch1DecodeError` (never returns a
 * partial/garbage result) on: missing/wrong magic, a truncated header,
 * string table, or record, a byte-length mismatch against the declared
 * chunk/string counts, invalid UTF-8, or a string-table index a record
 * points at that doesn't exist -- the exact same checks
 * src/spatial/chunks.py's `deserialize_chunk` performs server-side. */
export function decodeFvch1(buffer: ArrayBuffer): DecodedChunkSet {
  if (buffer.byteLength < HEADER_SIZE) {
    throw new Fvch1DecodeError(`too short to contain a header (${buffer.byteLength} bytes)`);
  }
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer, 0, 6);
  for (let i = 0; i < 6; i++) {
    if (bytes[i] !== MAGIC[i]) throw new Fvch1DecodeError("missing FVCH1 magic/version header");
  }

  let offset = 6;
  const chunkCount = view.getUint32(offset, true);
  offset += 4;
  const stringCount = view.getUint32(offset, true);
  offset += 4;

  const stringTable: string[] = new Array(stringCount);
  const decoder = new TextDecoder("utf-8", { fatal: true });
  for (let i = 0; i < stringCount; i++) {
    if (offset + 2 > buffer.byteLength) throw new Fvch1DecodeError("string table truncated (length prefix cut off)");
    const len = view.getUint16(offset, true);
    offset += 2;
    if (offset + len > buffer.byteLength) throw new Fvch1DecodeError("string table truncated (string data cut off)");
    try {
      stringTable[i] = decoder.decode(new Uint8Array(buffer, offset, len));
    } catch {
      throw new Fvch1DecodeError("invalid UTF-8 in string table");
    }
    offset += len;
  }

  const expectedTotal = offset + chunkCount * RECORD_SIZE;
  if (expectedTotal !== buffer.byteLength) {
    throw new Fvch1DecodeError(
      `expected ${expectedTotal} bytes for ${chunkCount} chunk(s) + header/string-table, got ${buffer.byteLength}`
    );
  }

  if (chunkCount === 0) return { ...emptySet(), stringTable };

  const set = emptySet();
  set.count = chunkCount;
  set.level = new Uint8Array(chunkCount);
  set.isLeaf = new Uint8Array(chunkCount);
  set.bandId = new Int32Array(chunkCount);
  set.ix = new Int32Array(chunkCount);
  set.iy = new Int32Array(chunkCount);
  set.cellSizeM = new Float32Array(chunkCount);
  set.xMin = new Float32Array(chunkCount);
  set.xMax = new Float32Array(chunkCount);
  set.yMin = new Float32Array(chunkCount);
  set.yMax = new Float32Array(chunkCount);
  set.centerX = new Float32Array(chunkCount);
  set.centerY = new Float32Array(chunkCount);
  set.rangeM = new Float32Array(chunkCount);
  set.pointCount = new Uint32Array(chunkCount);
  set.elevationMin = new Float32Array(chunkCount);
  set.elevationMax = new Float32Array(chunkCount);
  set.elevationMedian = new Float32Array(chunkCount);
  set.semanticPriority = new Float32Array(chunkCount);
  set.majorityClassFraction = new Float32Array(chunkCount);
  set.geometryScore = new Float32Array(chunkCount);
  set.dynamicScore = new Float32Array(chunkCount);
  set.informationScore = new Float32Array(chunkCount);
  set.semanticClassIdx = new Uint16Array(chunkCount);
  set.terrainClassIdx = new Uint16Array(chunkCount);
  set.obstacleClassIdx = new Uint16Array(chunkCount);
  set.resolutionLabelIdx = new Uint16Array(chunkCount);
  set.labelSourceIdx = new Uint16Array(chunkCount);
  set.sourceIdx = new Uint16Array(chunkCount);
  set.stringTable = stringTable;

  const checkIdx = (idx: number, field: string, record: number) => {
    if (idx < 0 || idx >= stringTable.length) {
      throw new Fvch1DecodeError(`${field} string index ${idx} out of range at record ${record} (table size ${stringTable.length})`);
    }
  };

  for (let i = 0; i < chunkCount; i++) {
    if (offset + RECORD_SIZE > buffer.byteLength) throw new Fvch1DecodeError(`record ${i} truncated`);
    set.level[i] = view.getUint8(offset);
    set.isLeaf[i] = view.getUint8(offset + 1);
    set.bandId[i] = view.getInt32(offset + 2, true);
    set.ix[i] = view.getInt32(offset + 6, true);
    set.iy[i] = view.getInt32(offset + 10, true);
    set.cellSizeM[i] = view.getFloat32(offset + 14, true);
    set.xMin[i] = view.getFloat32(offset + 18, true);
    set.xMax[i] = view.getFloat32(offset + 22, true);
    set.yMin[i] = view.getFloat32(offset + 26, true);
    set.yMax[i] = view.getFloat32(offset + 30, true);
    set.centerX[i] = view.getFloat32(offset + 34, true);
    set.centerY[i] = view.getFloat32(offset + 38, true);
    set.rangeM[i] = view.getFloat32(offset + 42, true);
    set.pointCount[i] = view.getUint32(offset + 46, true);
    set.elevationMin[i] = view.getFloat32(offset + 50, true);
    set.elevationMax[i] = view.getFloat32(offset + 54, true);
    set.elevationMedian[i] = view.getFloat32(offset + 58, true);
    set.semanticPriority[i] = view.getFloat32(offset + 62, true);
    set.majorityClassFraction[i] = view.getFloat32(offset + 66, true);
    set.geometryScore[i] = view.getFloat32(offset + 70, true);
    set.dynamicScore[i] = view.getFloat32(offset + 74, true);
    set.informationScore[i] = view.getFloat32(offset + 78, true);

    const semIdx = view.getUint16(offset + 82, true);
    const terIdx = view.getUint16(offset + 84, true);
    const obsIdx = view.getUint16(offset + 86, true);
    const resIdx = view.getUint16(offset + 88, true);
    const lsIdx = view.getUint16(offset + 90, true);
    const srcIdx = view.getUint16(offset + 92, true);
    checkIdx(semIdx, "semantic_class", i);
    checkIdx(terIdx, "terrain_class", i);
    checkIdx(obsIdx, "obstacle_class", i);
    checkIdx(resIdx, "resolution_label", i);
    checkIdx(lsIdx, "label_source", i);
    checkIdx(srcIdx, "source", i);
    set.semanticClassIdx[i] = semIdx;
    set.terrainClassIdx[i] = terIdx;
    set.obstacleClassIdx[i] = obsIdx;
    set.resolutionLabelIdx[i] = resIdx;
    set.labelSourceIdx[i] = lsIdx;
    set.sourceIdx[i] = srcIdx;

    offset += RECORD_SIZE;
  }

  return set;
}

/** One chunk's fields resolved into a plain object -- built on demand (e.g.
 * after a click), never for the whole set at once. */
export interface ChunkInfo {
  nodeId: string;
  level: number;
  resolutionLabel: string;
  isLeaf: boolean;
  cellSizeM: number;
  centerX: number;
  centerY: number;
  rangeM: number;
  pointCount: number;
  elevationMin: number;
  elevationMax: number;
  elevationMedian: number;
  semanticClass: string;
  semanticPriority: number;
  majorityClassFraction: number;
  geometryScore: number;
  dynamicScore: number;
  informationScore: number;
  terrainClass: string;
  obstacleClass: string;
  labelSource: string;
  source: string;
}

export function chunkInfoAt(set: DecodedChunkSet, i: number): ChunkInfo {
  return {
    nodeId: `${set.level[i]}:${set.bandId[i]}:${set.ix[i]}:${set.iy[i]}`,
    level: set.level[i],
    resolutionLabel: set.stringTable[set.resolutionLabelIdx[i]],
    isLeaf: set.isLeaf[i] === 1,
    cellSizeM: set.cellSizeM[i],
    centerX: set.centerX[i],
    centerY: set.centerY[i],
    rangeM: set.rangeM[i],
    pointCount: set.pointCount[i],
    elevationMin: set.elevationMin[i],
    elevationMax: set.elevationMax[i],
    elevationMedian: set.elevationMedian[i],
    semanticClass: set.stringTable[set.semanticClassIdx[i]],
    semanticPriority: set.semanticPriority[i],
    majorityClassFraction: set.majorityClassFraction[i],
    geometryScore: set.geometryScore[i],
    dynamicScore: set.dynamicScore[i],
    informationScore: set.informationScore[i],
    terrainClass: set.stringTable[set.terrainClassIdx[i]],
    obstacleClass: set.stringTable[set.obstacleClassIdx[i]],
    labelSource: set.stringTable[set.labelSourceIdx[i]],
    source: set.stringTable[set.sourceIdx[i]],
  };
}

/** Decodes a base64 string (as returned by ChunksResponse.payload_base64)
 * into an ArrayBuffer -- browser-native, no dependency. */
export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
