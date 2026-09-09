export {
  MergeManifest,
  ToolMerger,
  PAPER_2026_DEFAULTS as PAPER_2026_MERGER_DEFAULTS,
  PAPER_2026_MERGER_RESOURCE_DEFAULTS,
  type CandidatePair,
  type ClusterValidation,
  type ClusterValidator,
  type DescriptorSynthesizer,
  type MergeManifestEntry,
  type MergedToolDescriptor,
  type MergeResult,
  type RelationshipClassification,
  type RelationshipClassifier,
  type SynthesizedDescriptor,
  type ToolMergerOptions,
} from "./merger.js";
export {
  buildCorrectionPrompt,
  buildCorrectionPrompt as buildClusterCorrectionPrompt,
  buildDescriptorSynthesisPrompt,
  buildQueryDecompositionPrompt,
  buildRelationshipPrompt,
} from "./prompts.js";
export * from "./retriever.js";

import { PAPER_2026_DEFAULTS as mergerDefaults } from "./merger.js";
import { PAPER_2026_RETRIEVER_DEFAULTS as retrieverDefaults } from "./retriever.js";

/** Paper reproduction-oriented algorithm defaults. Model choices remain caller-provided. */
export const PAPER_2026_DEFAULTS = Object.freeze({
  ...mergerDefaults,
  ...retrieverDefaults,
});

/** Name matching the ACL paper while retaining the explicit implementation class. */
export { ToolMerger as ToolScopeMerger } from "./merger.js";
/** Name matching the ACL paper while retaining the explicit implementation class. */
export { MultiQueryRetriever as ToolScopeRetriever } from "./retriever.js";
