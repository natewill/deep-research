import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import { generateText } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import { mdToPdf } from 'md-to-pdf';
import fs from 'fs';

import {
  o3MiniModel,
  trimPrompt,
  gpt4Model,
  gpt4MiniModel,
  deepseekR1,
  groqR1,
  gemini,
  togetherR1,
  infraR1,
} from './ai/providers';
import { systemPrompt } from './prompt';

const curr_model = gemini;

type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

const ConcurrencyLimit = 2;

const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

// helper to safely generate json from generateText output
async function safeGenerate<T>({
  model,
  system,
  prompt,
  schema,
  abortSignal,
}: {
  model: any;
  system: string;
  prompt: string;
  schema: z.Schema<T>;
  abortSignal?: AbortSignal;
}): Promise<T> {
  const response = await generateText({ model, system, prompt, abortSignal });
  const responseText = response.text ?? '';

  // extract json using regex
  const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/i);
  const jsonString = jsonMatch && jsonMatch[1] ? jsonMatch[1].trim() : responseText.trim();

  console.log("raw json response:", jsonString);

  try {
    // sanitize json (removes control characters & fixes common issues)
    const sanitizedJson = jsonString
      .replace(/[\u0000-\u001F\u007F]/g, '') // removes bad control chars
      .replace(/,(\s*[}\]])/g, '$1'); // removes trailing commas

    const parsed = JSON.parse(sanitizedJson);
    return schema.parse(parsed);
  } catch (e) {
    console.error("failed to parse json:", jsonString);
    throw new Error(`invalid json from model: ${jsonString}`);
  }
}

async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
}) {
  // wait 15 seconds before calling groq
  await new Promise(resolve => setTimeout(resolve, 30000));

  const prompt = `given the following prompt from the user, generate a list of serp queries to research the topic.

### output format:
return your response in **valid json format**, following this **exact** structure:
\`\`\`json
{
  "queries": [
    {
      "query": "string",
      "researchGoal": "string"
    }
  ]
}
\`\`\`

### rules:
- **only return valid json.** no explanations or extra text.
- each query **must be unique** and not redundant.
- each research goal should be **detailed and suggest further research directions**.

user prompt:
<prompt>${query}</prompt>

${learnings ? `here are previous research learnings to refine queries: ${learnings.join('\n')}` : ''}
`;

  const schema = z.object({
    queries: z.array(
      z.object({
        query: z.string(),
        researchGoal: z.string(),
      })
    ),
  });

  const res = await safeGenerate({
    model: curr_model,
    system: systemPrompt(),
    prompt,
    schema,
  });

  console.log(`created ${res.queries.length} queries`, res.queries);
  return res.queries.slice(0, numQueries);
}

async function processSerpResult({
  query,
  result,
  numLearnings = 3,
  numFollowUpQuestions = 3,
}: {
  query: string;
  result: SearchResponse;
  numLearnings?: number;
  numFollowUpQuestions?: number;
}) {
  const contents = compact(result.data.map(item => item.description)).map(content =>
    trimPrompt(content, 25000)
  );
  console.log(`ran ${query}, found ${contents.length} contents`);

  // wait 15 seconds before calling groq
  await new Promise(resolve => setTimeout(resolve, 30000));

  const prompt = `given the following contents from a serp search for the query <query>${query}</query>, generate a structured json response with two fields: "learnings" and "followUpQuestions". 

return your response in the following json format:
\`\`\`json
{
  "learnings": ["string1", "string2", "string3"],
  "followUpQuestions": ["question1", "question2", "question3"]
}
\`\`\`

make sure:
- the json is well-formed.
- each learning is unique and contains specific details, numbers, or entities.
- the follow-up questions help refine further research.

<contents>${contents
    .map(content => `<content>\n${content}\n</content>`)
    .join('\n')}</contents>
`;

  const schema = z.object({
    learnings: z.array(z.string()),
    followUpQuestions: z.array(z.string()),
  });

  const res = await safeGenerate({
    model: curr_model,
    system: systemPrompt(),
    prompt,
    schema,
    abortSignal: AbortSignal.timeout(60000),
  });

  console.log(`created ${res.learnings.length} learnings`, res.learnings);
  return res;
}

export async function writeFinalReportPdf({
  prompt,
  learnings,
  visitedUrls,
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}): Promise<string> {
  const learningsString = trimPrompt(
    learnings.map(l => `<learning>\n${l}\n</learning>`).join('\n'),
    150000
  );

  // wait before calling the model
  await new Promise(resolve => setTimeout(resolve, 30000));

  const promptText = `given the following prompt from the user, write a final academic report on the topic using the learnings from research. make it as detailed as possible, aim for 5 or more pages, and include all research learnings.

### output format:
return your response in **valid json format**, following this exact structure:
\`\`\`json
{
  "reportMarkdown": "string"
}
\`\`\`

- **only return valid json.** no extra explanations or text.

### user prompt:
<prompt>${prompt}</prompt>

### research learnings:
<learnings>
${learningsString}
</learnings>
`;

  const schema = z.object({
    reportMarkdown: z.string(),
  });

  const res = await safeGenerate({
    model: curr_model,
    system: systemPrompt(),
    prompt: promptText,
    schema,
  });

  const urlsSection = `\n\n## sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
  const reportMarkdown = res.reportMarkdown + urlsSection;

  // custom css for academic paper styling
  const css = `
  @page {
    size: a4;
    margin: 2cm;
  }
  body {
    font-family: "times new roman", serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #000;
    margin: 0;
    padding: 0;
  }
  h1, h2, h3, h4, h5, h6 {
    font-weight: bold;
    text-align: center;
    margin: 1em 0 0.5em 0;
  }
  h1 {
    font-size: 24pt;
  }
  h2 {
    font-size: 20pt;
  }
  h3 {
    font-size: 16pt;
  }
  p {
    text-align: justify;
    margin: 1em 0;
  }
  .title-page {
    text-align: center;
    margin-top: 200px;
  }
  .title-page h1 {
    font-size: 28pt;
  }
  .title-page h2 {
    font-size: 18pt;
    font-weight: normal;
  }
  footer {
    position: fixed;
    bottom: 1cm;
    text-align: center;
    font-size: 10pt;
    color: #666;
  }
  `;

  const options = { stylesheet: css };
  const pdfResult = await mdToPdf({ content: reportMarkdown, ...options });
  const outputPath = './final-report.pdf';
  fs.writeFileSync(outputPath, pdfResult.content);
  console.log(`pdf saved to ${outputPath}`);
  return outputPath;
}

export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
}): Promise<ResearchResult> {
  const serpQueries = await generateSerpQueries({
    query,
    learnings,
    numQueries: breadth,
  });

  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          // wait 15 seconds before calling firecrawl search
          await new Promise(resolve => setTimeout(resolve, 30000));

          if (!serpQuery?.query) {
            console.error("invalid query: ", serpQuery);
            return { learnings: [], visitedUrls: [] };
          }

          const result = await firecrawl.search(serpQuery.query, {
            timeout: 15000,
            limit: 5,
          });

          // collect urls from this search
          const newUrls = compact(result.data.map(item => item.url));
          const newBreadth = Math.ceil(breadth / 2);
          const newDepth = depth - 1;

          const newLearnings = await processSerpResult({
            query: serpQuery.query,
            result,
            numFollowUpQuestions: newBreadth,
          });
          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          if (newDepth > 0) {
            console.log(`researching deeper, breadth: ${newBreadth}, depth: ${newDepth}`);

            const nextQuery = `
            previous research goal: ${serpQuery.researchGoal}
            follow-up research directions: ${newLearnings.followUpQuestions.map(q => `\n${q}`).join('')}
          `.trim();

            return deepResearch({
              query: nextQuery,
              breadth: newBreadth,
              depth: newDepth,
              learnings: allLearnings,
              visitedUrls: allUrls,
            });
          } else {
            return {
              learnings: allLearnings,
              visitedUrls: allUrls,
            };
          }
        } catch (e) {
          console.error(`error running query: ${serpQuery.query}: `, e);
          return { learnings: [], visitedUrls: [] };
        }
      })
    )
  );

  return {
    learnings: [...new Set(results.flatMap(r => r.learnings))],
    visitedUrls: [...new Set(results.flatMap(r => r.visitedUrls))],
  };
}