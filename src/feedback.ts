import { generateText } from 'ai';
import { z } from 'zod';
import { gemini } from './ai/providers';
import { systemPrompt } from './prompt';

const curr_model = gemini;

export async function generateFeedback({
  query,
  numQuestions = 3,
}: {
  query: string;
  numQuestions?: number;
}) {
  const prompt = `given the following query from the user, ask some follow-up questions to clarify the research direction.

return your response in **valid json format**, following this exact structure:
\`\`\`json
{
  "questions": ["question1", "question2", "question3"]
}
\`\`\`

### rules:
- only return valid json. no extra text or explanations.
- ensure the response is well-formed json.
- each question must be concise and relevant.

user query:
<query>${query}</query>
`;

  try {
    const response = await generateText({
      model: curr_model,
      system: systemPrompt(),
      prompt,
    });

    const responseText = response.text ?? ''; // ensure response.text is never undefined

    // extract json from markdown code block, if present
    const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/i);
    const jsonString = jsonMatch && jsonMatch[1] ? jsonMatch[1].trim() : responseText.trim();

    const parsed = JSON.parse(jsonString);
    const schema = z.object({
      questions: z.array(z.string()),
    });
    const validated = schema.parse(parsed);
    return validated.questions.slice(0, numQuestions);
  } catch (error) {
    console.error('error parsing gemini response:', error);
    return [];
  }
}