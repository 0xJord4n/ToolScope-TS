import { describe, expect, test } from "bun:test";
import {
  MultiQueryRetriever,
  PAPER_2026_DEFAULTS,
  ToolMerger,
  buildClusterCorrectionPrompt,
  buildDescriptorSynthesisPrompt,
  buildQueryDecompositionPrompt,
  buildRelationshipPrompt,
} from "../src/paper/index.js";

describe("paper pipeline public entrypoint", () => {
  test("exports the complete opt-in pipeline surface", () => {
    expect(ToolMerger).toBeFunction();
    expect(MultiQueryRetriever).toBeFunction();
    expect(PAPER_2026_DEFAULTS).toMatchObject({
      candidateNeighbors: 30,
      candidateThreshold: 0.82,
      autoCorrectionPasses: 1,
      alpha: 1,
      rerankPoolSize: 50,
      epsilon: 1e-12,
    });
    expect(buildRelationshipPrompt).toBeFunction();
    expect(buildClusterCorrectionPrompt).toBeFunction();
    expect(buildDescriptorSynthesisPrompt).toBeFunction();
    expect(buildQueryDecompositionPrompt).toBeFunction();
  });
});
