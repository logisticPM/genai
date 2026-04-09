import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const TAXONOMY = `
- misrepresentation: bar/area sizes do not match labeled values
- 3d: 3D effects distort visual comparison
- truncated axis: y-axis doesn't start at zero, exaggerating differences
- inappropriate use of pie chart: used for data unsuitable for part-to-whole comparison
- inconsistent tick intervals: axis ticks are unevenly spaced
- dual axis: two y-axes with different scales mislead comparisons
- inconsistent binning size: histogram bins have unequal widths without normalization
- discretized continuous variable: continuous data binned to hide distribution
- inappropriate use of line chart: used for non-sequential or categorical data
- inappropriate item order: ordering creates false impressions
- inverted axis: axis direction reversed, flipping perceived trend
- inappropriate axis range: range set to exaggerate or minimize differences
`.trim();

export async function POST(req: NextRequest) {
  const { imageBase64, apiKey, context } = await req.json();

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 400 });
  }

  const client = new OpenAI({ apiKey, baseURL: "https://openrouter.ai/api/v1" });

  const prompt = context
    ? `You are an expert in data visualization. Detect misleading elements by comparing the chart against the provided data context.\n\n## Misleader Taxonomy\n${TAXONOMY}\n\n## Ground-Truth / Context Data\n${context}\n\n## Output\nRespond with valid JSON only:\n{\n  "misleading": <true|false>,\n  "misleader_types": [<zero or more types from the taxonomy>],\n  "explanation": "<two to three sentences>"\n}`
    : `You are an expert in data visualization. Detect misleading elements in the chart image.\n\n## Misleader Taxonomy\n${TAXONOMY}\n\n## Output\nRespond with valid JSON only:\n{\n  "misleading": <true|false>,\n  "misleader_types": [<zero or more types from the taxonomy>],\n  "explanation": "<two to three sentences>"\n}`;

  try {
    const response = await client.chat.completions.create({
      model: "anthropic/claude-haiku-4.5",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
          ],
        },
      ],
      max_tokens: 512,
    });

    let raw = response.choices[0].message.content?.trim() ?? "";
    if (raw.startsWith("```")) {
      raw = raw.split("```")[1];
      if (raw.startsWith("json")) raw = raw.slice(4);
    }
    const result = JSON.parse(raw);
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
