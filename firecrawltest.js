
import FirecrawlApp from '@mendable/firecrawl-js';
import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' }); // explicitly load

// increase this if you have higher API rate limits
const ConcurrencyLimit = 2;

// Initialize Firecrawl with optional API key and optional base url

const firecrawl = new FirecrawlApp({
    apiKey: process.env.FIRECRAWL_KEY,
    apiUrl: process.env.FIRECRAWL_BASE_URL,
  });

const result = await firecrawl.search("USAID funding mechanisms", {
    timeout: 15000,
    limit: 5,
  });
  import { compact } from 'lodash-es';

  const newUrls = compact(result.data.map(item => item.url));

  console.log(newUrls);