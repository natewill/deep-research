import { createOpenAI } from '@ai-sdk/openai';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { getEncoding } from 'js-tiktoken';
import { createGroq } from '@ai-sdk/groq';
import { createTogetherAI } from '@ai-sdk/togetherai';
import { RecursiveCharacterTextSplitter } from './text-splitter';
import { createDeepInfra } from '@ai-sdk/deepinfra';

// Providers




const google = createGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
});

const deepseek = createDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const togetherai = createTogetherAI({
  apiKey: process.env.TOGETHER_AI_API_KEY,
});

const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
})

const openai = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
});

const deepinfra = createDeepInfra({
  apiKey: process.env.DEEPINFRA_API_KEY,
});


// Models

export const groqR1 = groq('llama-3.3-70b-versatile', {

})

export const infraR1 = deepinfra('deepseek-ai/DeepSeek-R1')

export const togetherR1 = togetherai('deepseek-ai/DeepSeek-R1-Distill-Llama-70B');

export const gemini = google('gemini-2.0-flash-thinking-exp-01-21', {
  structuredOutputs: false,
})

export const deepseekR1 = deepseek('deepseek-reasoner', {

});

export const gpt4Model = openai('gpt4o', {
  structuredOutputs: true,
});
export const gpt4MiniModel = openai('gpt-4o-mini', {
  structuredOutputs: true,
});
export const o3MiniModel = openai('o3-mini', {
  reasoningEffort: 'medium',
  structuredOutputs: true,
});

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(prompt: string, contextSize = 120_000) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
