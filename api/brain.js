// IMBOT Cloud Engine — 16kHz u-Law Studio Audio + Whisper V3 + 2026 Smart Search + Dual AI

const parseKeys = (headerVal) =>
  String(headerVal || '')
    .split(',')
    .map((k) => k.trim())
    .filter((k) => k.length > 8);

function getDhakaTimeInfo() {
  const now = new Date();
  const dhakaMs = now.getTime() + 6 * 60 * 60 * 1000;
  const d = new Date(dhakaMs);
  const pad = (n) => String(n).padStart(2, '0');
  const yyyy = d.getUTCFullYear();
  const mm = pad(d.getUTCMonth() + 1);
  const dd = pad(d.getUTCDate());
  const hh = pad(d.getUTCHours());
  const min = pad(d.getUTCMinutes());
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const tomorrow = new Date(dhakaMs + 24 * 60 * 60 * 1000);
  const tomStr = `${tomorrow.getUTCFullYear()}-${pad(tomorrow.getUTCMonth() + 1)}-${pad(tomorrow.getUTCDate())}`;
  return {
    year: yyyy,
    dateStr: `${yyyy}-${mm}-${dd}`,
    tomorrowStr: tomStr,
    time24: `${hh}:${min}`,
    dayName: days[d.getUTCDay()],
    full: `${days[d.getUTCDay()]}, ${yyyy}-${mm}-${dd} ${hh}:${min} (Dhaka Time, UTC+6)`
  };
}

async function readRawBody(req, isAudio) {
  if (Buffer.isBuffer(req.body)) return req.body;
  if (req.body instanceof Uint8Array) return Buffer.from(req.body);
  if (typeof req.body === 'string') return Buffer.from(req.body, isAudio ? 'latin1' : 'utf8');
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// G.711 u-Law -> 16-bit Linear PCM + স্মার্ট RMS অডিও নরমালাইজেশন
function muLawToLinear(uVal) {
  uVal = (~uVal) & 0xFF;
  const sign = uVal & 0x80;
  const exponent = (uVal >> 4) & 0x07;
  const mantissa = uVal & 0x0F;
  let sample = ((mantissa << 3) + 0x84) << exponent;
  sample -= 0x84;
  return sign ? -sample : sample;
}

function normalizeToCleanPcmWav(rawBuf) {
  if (!rawBuf || rawBuf.length < 48 || rawBuf.toString('ascii', 0, 4) !== 'RIFF') {
    return rawBuf;
  }

  const audioFormat = rawBuf.readUInt16LE(20);
  const sampleRate = rawBuf.readUInt32LE(24) || 16000;
  const dataOffset = 44;
  const payloadLen = rawBuf.length - dataOffset;
  if (payloadLen <= 0) return rawBuf;

  let samples;
  if (audioFormat === 7) {
    // 16kHz u-Law (1 byte/sample) -> 16-bit PCM
    samples = new Int16Array(payloadLen);
    for (let i = 0; i < payloadLen; i++) {
      samples[i] = muLawToLinear(rawBuf[dataOffset + i]);
    }
  } else if (audioFormat === 1) {
    const cnt = Math.floor(payloadLen / 2);
    samples = new Int16Array(cnt);
    for (let i = 0; i < cnt; i++) {
      samples[i] = rawBuf.readInt16LE(dataOffset + i * 2);
    }
  } else {
    return rawBuf;
  }

  // প্রথম ৫০ms এর টাচ-ক্লিক বাদ দিয়ে RMS বের করা (যাতে ভয়েস চাপা না পড়ে)
  const skip = Math.min(samples.length, Math.floor(sampleRate * 0.05));
  let sumSq = 0;
  let activeCnt = 0;
  for (let i = skip; i < samples.length; i++) {
    const v = samples[i];
    if (Math.abs(v) > 120) {
      sumSq += v * v;
      activeCnt++;
    }
  }

  if (activeCnt > 200) {
    const rms = Math.sqrt(sumSq / activeCnt);
    if (rms > 60) {
      let gain = 4600 / rms;
      if (gain > 8.0) gain = 8.0;
      if (gain < 0.7) gain = 0.7;
      for (let i = 0; i < samples.length; i++) {
        // সফট লিমিটার (ক্লিপিং ছাড়াই স্পষ্ট ভয়েস)
        const scaled = (samples[i] * gain) / 30000;
        const soft = Math.tanh(scaled) * 30000;
        samples[i] = Math.max(-32767, Math.min(32767, Math.round(soft)));
      }
    }
  }

  const pcmBytes = samples.length * 2;
  const out = Buffer.alloc(44 + pcmBytes);
  out.write('RIFF', 0, 'ascii');
  out.writeUInt32LE(36 + pcmBytes, 4);
  out.write('WAVE', 8, 'ascii');
  out.write('fmt ', 12, 'ascii');
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * 2, 28);
  out.writeUInt16LE(2, 32);
  out.writeUInt16LE(16, 34);
  out.write('data', 36, 'ascii');
  out.writeUInt32LE(pcmBytes, 40);
  for (let i = 0; i < samples.length; i++) {
    out.writeInt16LE(samples[i], 44 + i * 2);
  }
  return out;
}

// ১. Groq Whisper Large V3 (বাংলার জন্য সর্বোচ্চ অ্যাকুরেট)
async function transcribeWithGroq(cleanWavBuf, groqKeys) {
  const groq429 = [];
  const models = ['whisper-large-v3', 'whisper-large-v3-turbo'];

  for (let i = 0; i < groqKeys.length; i++) {
    for (const model of models) {
      try {
        const form = new FormData();
        form.append('file', new Blob([cleanWavBuf], { type: 'audio/wav' }), 'audio.wav');
        form.append('model', model);
        form.append('language', 'bn');
        form.append('temperature', '0');
        form.append(
          'prompt',
          'ইমবট, আল ইমরান, অ্যালার্ম, টাইমার, ফোকাস, গেম, টিক ট্যাক টো, রক পেপার সিজার, ডাইনো, ফ্ল্যাপি, কয়েন টস, ভলিউম, আবহাওয়া, সময়, বর্তমান প্রধানমন্ত্রী, তারেক রহমান, আইফোন ১৭ প্রো ম্যাক্স, ঢাকা, বাংলাদেশ।'
        );

        const resp = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
          method: 'POST',
          headers: { Authorization: `Bearer ${groqKeys[i]}` },
          body: form,
          signal: AbortSignal.timeout(8500)
        });

        if (resp.status === 429) {
          if (!groq429.includes(i)) groq429.push(i);
          break;
        }
        if (!resp.ok) continue;

        const data = await resp.json();
        const text = (data && data.text ? String(data.text) : '').trim();
        if (text && text.length > 1) {
          return { text, groqIdx: i, groq429 };
        }
      } catch (_) {
        continue;
      }
    }
  }
  return { text: '', groqIdx: -1, groq429 };
}

// ২. স্মার্ট লাইভ সার্চ ডিটেক্টর
function needsLiveSearch(text) {
  const t = String(text || '').trim();
  if (!t) return false;
  const skipLocal = /(অ্যালার্ম|এলার্ম|alarm|টাইমার|timer|ফোকাস|focus|ভলিউম|সাউন্ড|volume|গেম|game|টিক ট্যাক|রক পেপার|ডাইনো|ফ্ল্যাপি|কয়েন|কয়েন|ঘুমা|sleep|ইন্ট্রো|intro|তোমার নাম|কে বানিয়েছে|বানাইছে|নির্মাতা|মেকার|ধাঁধা|গল্প|জোকস|কৌতুক)/i;
  if (skipLocal.test(t)) return false;

  const liveTriggers = /(আজ|আজকে|এখন|বর্তমান|সর্বশেষ|লেটেস্ট|নিউজ|খবর|দাম|প্রাইস|টাকা|প্রধানমন্ত্রী|রাষ্ট্রপতি|উপদেষ্টা|সরকার|নির্বাচন|খেলা|ম্যাচ|স্কোর|ক্রিকেট|ফুটবল|বিশ্বকাপ|আইফোন|iphone|samsung|ডলার|সোনা|স্বর্ণ|বাজারে|নতুন|আপডেট|মারা|বেঁচে|বাজেট|2025|2026|2027|১৭|17|price|latest|news|current|today|now|prime minister|president|score|match|winner|who is)/i;
  return liveTriggers.test(t);
}

// ৩. বাংলা প্রশ্ন -> ২০২৬-টার্গেটেড ইংরেজি সার্চ কোয়েরি (~150ms)
async function build2026SearchQuery(userText, timeInfo, groqKeys, existing429) {
  for (let i = 0; i < groqKeys.length; i++) {
    if (existing429.includes(i)) continue;
    try {
      const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${groqKeys[i]}`
        },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          messages: [
            {
              role: 'system',
              content: `Today is ${timeInfo.dateStr} (year ${timeInfo.year}). Convert the user's question into ONE short English web search query (max 8 words) including "${timeInfo.year}" to get the latest current fact. For Bangladesh topics include "Bangladesh". Output ONLY the query text.`
            },
            { role: 'user', content: userText }
          ],
          max_tokens: 30,
          temperature: 0
        }),
        signal: AbortSignal.timeout(2500)
      });
      if (!resp.ok) continue;
      const data = await resp.json();
      const q = String(data?.choices?.[0]?.message?.content || '')
        .replace(/["'\n]/g, '')
        .trim();
      if (q.length >= 4) return q;
    } catch (_) {
      continue;
    }
  }
  return `${userText} ${timeInfo.year} latest`;
}

// ৪. লাইভ ওয়েব সার্চ (Tavily + আনলিমিটেড ফ্রি Google News RSS ব্যাকআপ)
async function runLiveSearch(searchQuery, rawBangla, timeInfo, tavilyKeys) {
  const tav429 = [];

  for (let i = 0; i < tavilyKeys.length; i++) {
    try {
      const resp = await fetch('https://api.tavily.com/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: tavilyKeys[i],
          query: searchQuery,
          search_depth: 'basic',
          max_results: 5,
          include_answer: true
        }),
        signal: AbortSignal.timeout(5000)
      });

      if (resp.status === 429 || resp.status === 432) {
        tav429.push(i);
        continue;
      }
      if (!resp.ok) continue;

      const data = await resp.json();
      let summary = `Search Query Used: ${searchQuery}\n`;
      if (data.answer) summary += `Direct Answer (${timeInfo.year}): ${data.answer}\n`;
      if (Array.isArray(data.results)) {
        for (const r of data.results.slice(0, 4)) {
          summary += `- ${r.title}: ${String(r.content || '').slice(0, 260)}\n`;
        }
      }
      if (summary.length > 40) {
        return { context: summary.trim(), source: 'tavily', tavIdx: i, tav429 };
      }
    } catch (_) {
      continue;
    }
  }

  // ফ্রি আনলিমিটেড ব্যাকআপ: Google News RSS (ইংরেজি ২০২৬ + বাংলা)
  try {
    const rssUrls = [
      `https://news.google.com/rss/search?q=${encodeURIComponent(searchQuery + ' when:90d')}&hl=en-US&gl=BD&ceid=BD:en`,
      `https://news.google.com/rss/search?q=${encodeURIComponent(rawBangla + ' ' + timeInfo.year)}&hl=bn&gl=BD&ceid=BD:bn`
    ];
    for (const url of rssUrls) {
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
        signal: AbortSignal.timeout(4000)
      });
      if (!resp.ok) continue;
      const xml = await resp.text();
      const items = [...xml.matchAll(/<item>[\s\S]*?<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>[\s\S]*?<pubDate>([\s\S]*?)<\/pubDate>[\s\S]*?<\/item>/g)];
      if (items.length > 0) {
        const lines = items.slice(0, 6).map((m) => {
          const title = m[1].replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;/g, "'").trim();
          const pub = m[2].trim();
          return `- ${title} (${pub})`;
        });
        return { context: lines.join('\n'), source: 'rss', tavIdx: -1, tav429 };
      }
    }
  } catch (_) {}

  return { context: '', source: 'none', tavIdx: -1, tav429 };
}

// ৫. সিস্টেম প্রম্পট
function buildSystemPrompt(timeInfo, status, searchContext) {
  return `You are IMBOT (ইমবট), a smart, accurate Bangla desktop AI robot created by Al Imran (আল ইমরান).

IDENTITY & TONE RULES:
1) Your name is ইমবট (IMBOT). If asked who made/built/created you, answer: আমাকে আল ইমরান বানিয়েছেন।
2) Always speak in natural, friendly, conversational Bangla using "তুমি / বলো". NEVER use formal "আপনি / বলুন" and NEVER use the words "বাডি" or "বন্ধু".
3) Be accurate and direct for commands and factual questions. For stories, riddles (ধাঁধা), or jokes, be lively and complete them fully.
4) Never output markdown symbols (*, #, **, backticks). Output plain clean Bangla speech text after the tag.
5) Current Dhaka Date & Time: ${timeInfo.full}. Today=${timeInfo.dateStr}, Tomorrow=${timeInfo.tomorrowStr}, Current Year=${timeInfo.year}.
6) Device Status: City=${status.city}, Weather=${status.weather}, Battery=${status.bat}, Volume=${status.vol}%.
${searchContext ? `\nCRITICAL LIVE WEB SEARCH RESULTS (${timeInfo.year}):\nYour pre-training knowledge is outdated (before ${timeInfo.year}). You MUST strictly trust and state the latest ${timeInfo.year} facts from these search results (for example, current elected Prime Minister, latest phone prices, current news) and NEVER repeat outdated 2024/2025 facts that contradict these results:\n${searchContext}\n` : ''}
COMMAND TAGS (Put the EXACT tag at the very start ONLY when the user gives a command, followed by a brief 3-7 word friendly Bangla confirmation):
- [ACT:ALARM:YYYY-MM-DD:HH:MM:ONCE] set one-time alarm on specific date (24h HH:MM). Use Today=${timeInfo.dateStr} for আজ/আজকে, Tomorrow=${timeInfo.tomorrowStr} for কাল/কালকে, or exact date for "৩০ তারিখ". সকাল=AM, দুপুর/বিকাল/সন্ধ্যা/রাত=PM (রাত ১২টা=00:00).
- [ACT:ALARM:DAILY:HH:MM:DAILY] set repeating daily alarm ONLY if user says প্রতিদিন/রোজ/daily.
- [ACT:ALARM:OFF] cancel all alarms.
- [ACT:ALARM:DEL:HH:MM] delete a specific alarm time.
- [ACT:VOL:UP] increase volume, [ACT:VOL:DOWN] decrease volume, [ACT:VOL:MAX] max volume, [ACT:VOL:LOW] low volume.
- [ACT:GAMES] open games menu, [ACT:GAME:TTT] Tic-Tac-Toe, [ACT:GAME:RPS] Rock Paper Scissors, [ACT:GAME:DINO] Dino Run, [ACT:GAME:FLAPPY] Flappy, [ACT:GAME:COIN] Coin Toss.
- [ACT:CLOCK] show/tell time, [ACT:WEATHER] show/tell weather, [ACT:INTRO] play intro.
- [ACT:TIMER:SECONDS] start timer in seconds, [ACT:TIMER:OFF] stop timer.
- [ACT:FOCUS:MINUTES] start focus (default 25), [ACT:FOCUS:OFF] stop focus.
- [ACT:TEXT:SHORT ENGLISH TEXT] scroll text on OLED.
- [ACT:SLEEP] go to sleep.

If NOT a command, start with ONE matching emotion tag: [HAPPY], [LAUGH], [LOVE], [SHY], [WINK], [AMAZED], [SHOCKED], [CONFUSED], [ANGRY], [SAD], [COOL], [HOT], [SHIVER], or [DIZZY], then give the Bangla answer directly.`;
}

// ৬. Gemini API
async function askGemini(sysPrompt, userText, wavBuffer, geminiKeys) {
  const gem429 = [];
  const models = ['gemini-flash-latest', 'gemini-flash-lite-latest'];

  for (let i = 0; i < geminiKeys.length; i++) {
    for (const model of models) {
      try {
        const parts = wavBuffer
          ? [
              { inline_data: { mime_type: 'audio/wav', data: wavBuffer.toString('base64') } },
              { text: sysPrompt }
            ]
          : [{ text: `${sysPrompt}\n\nUser says: ${userText}` }];

        const resp = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${geminiKeys[i]}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              contents: [{ parts }],
              generationConfig: { maxOutputTokens: 900, temperature: 0.3 }
            }),
            signal: AbortSignal.timeout(8500)
          }
        );

        if (resp.status === 429) {
          if (!gem429.includes(i)) gem429.push(i);
          break;
        }
        if (!resp.ok) continue;

        const data = await resp.json();
        const text = (data?.candidates?.[0]?.content?.parts || [])
          .map((p) => p.text || '')
          .join(' ')
          .replace(/\s+/g, ' ')
          .trim();

        if (text) return { reply: text, gemIdx: i, gem429 };
      } catch (_) {
        continue;
      }
    }
  }
  return { reply: '', gemIdx: -1, gem429 };
}

// ৭. Groq Llama 3.3 70B ব্যাকআপ AI
async function askGroqLLM(sysPrompt, userText, groqKeys, existing429) {
  const groq429 = [...existing429];
  for (let i = 0; i < groqKeys.length; i++) {
    if (groq429.includes(i)) continue;
    try {
      const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${groqKeys[i]}`
        },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: [
            { role: 'system', content: sysPrompt },
            { role: 'user', content: userText }
          ],
          max_tokens: 900,
          temperature: 0.3
        }),
        signal: AbortSignal.timeout(8000)
      });

      if (resp.status === 429) {
        groq429.push(i);
        continue;
      }
      if (!resp.ok) continue;

      const data = await resp.json();
      const text = String(data?.choices?.[0]?.message?.content || '')
        .replace(/\s+/g, ' ')
        .trim();

      if (text) return { reply: text, groqIdx: i, groq429 };
    } catch (_) {
      continue;
    }
  }
  return { reply: '', groqIdx: -1, groq429 };
}

async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Content-Type', 'application/json; charset=utf-8');

  if (req.method === 'OPTIONS') return res.status(200).end();

  const timeInfo = getDhakaTimeInfo();

  if (req.method === 'GET') {
    return res.status(200).send(
      JSON.stringify({
        ok: true,
        status: 'IMBOT Online',
        version: '2.1',
        dhaka_time: timeInfo.full
      })
    );
  }

  if (req.method !== 'POST') {
    return res.status(405).send(JSON.stringify({ ok: false, reply: 'Method Not Allowed' }));
  }

  try {
    const mode = String(req.headers['x-mode'] || 'text').toLowerCase();
    const isAudio = mode === 'audio';
    const geminiKeys = parseKeys(req.headers['x-gemini-keys']);
    const groqKeys = parseKeys(req.headers['x-groq-keys']);
    const tavilyKeys = parseKeys(req.headers['x-tavily-keys']);

    const status = {
      city: String(req.headers['x-city'] || 'Dhaka'),
      weather: String(req.headers['x-weather'] || 'Unknown'),
      bat: String(req.headers['x-bat'] || 'Unknown'),
      vol: String(req.headers['x-vol'] || '95')
    };

    const rawBody = await readRawBody(req, isAudio);
    let userText = '';
    let sttGroqIdx = -1;
    let groq429 = [];
    let fallbackWav = null;

    if (isAudio) {
      const cleanWav = normalizeToCleanPcmWav(rawBody);
      if (groqKeys.length > 0) {
        const stt = await transcribeWithGroq(cleanWav, groqKeys);
        userText = stt.text;
        sttGroqIdx = stt.groqIdx;
        groq429 = stt.groq429;
      }
      if (!userText && geminiKeys.length > 0) {
        fallbackWav = cleanWav;
      }
      if (!userText && !fallbackWav) {
        return res.status(200).send(
          JSON.stringify({
            ok: false,
            transcript: '',
            reply: '[CONFUSED] কথাটা পরিষ্কার শুনতে পাইনি, আরেকবার বলো তো।',
            engine: 'none',
            gemIdx: -1,
            groqIdx: sttGroqIdx,
            tavIdx: -1,
            search: 'none',
            gem429: [],
            groq429,
            tav429: []
          })
        );
      }
    } else {
      userText = rawBody.toString('utf8').trim();
    }

    // লাইভ ওয়েব সার্চ (২০২৬ কোয়েরি রিরাইটসহ)
    let searchRes = { context: '', source: 'none', tavIdx: -1, tav429: [] };
    if (userText && needsLiveSearch(userText)) {
      const smartQuery = await build2026SearchQuery(userText, timeInfo, groqKeys, groq429);
      searchRes = await runLiveSearch(smartQuery, userText, timeInfo, tavilyKeys);
    }

    const sysPrompt = buildSystemPrompt(timeInfo, status, searchRes.context);

    let reply = '';
    let engine = 'none';
    let gemIdx = -1;
    let gem429 = [];
    let finalGroqIdx = sttGroqIdx;

    if (geminiKeys.length > 0) {
      const g = await askGemini(sysPrompt, userText, fallbackWav, geminiKeys);
      reply = g.reply;
      gemIdx = g.gemIdx;
      gem429 = g.gem429;
      if (reply) engine = 'gemini';
    }

    if (!reply && userText && groqKeys.length > 0) {
      const q = await askGroqLLM(sysPrompt, userText, groqKeys, groq429);
      reply = q.reply;
      if (q.groqIdx !== -1) finalGroqIdx = q.groqIdx;
      groq429 = q.groq429;
      if (reply) engine = 'groq';
    }

    if (!reply) {
      reply = '[SAD] এপিআই কী-এর লিমিট শেষ বা সার্ভার ব্যস্ত, ওয়েব হাবে নতুন কী দাও।';
    }

    return res.status(200).send(
      JSON.stringify({
        ok: engine !== 'none',
        transcript: userText,
        reply,
        engine,
        gemIdx,
        groqIdx: finalGroqIdx,
        tavIdx: searchRes.tavIdx,
        search: searchRes.source,
        gem429,
        groq429,
        tav429: searchRes.tav429
      })
    );
  } catch (err) {
    return res.status(500).send(
      JSON.stringify({
        ok: false,
        transcript: '',
        reply: '[SAD] সার্ভার সাময়িক সমস্যায় আছে, আরেকবার চেষ্টা করো।',
        error: String(err && err.message ? err.message : err)
      })
    );
  }
}

module.exports = handler;
module.exports.config = { api: { bodyParser: false } };
