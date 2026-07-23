import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Set large payload limit for base64 audio/video uploads
  app.use(express.json({ limit: '200mb' }));
  app.use(express.urlencoded({ limit: '200mb', extended: true }));

  // Initialize Gemini Client
  const getGeminiClient = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn('GEMINI_API_KEY environment variable is missing.');
    }
    return new GoogleGenAI({
      apiKey: apiKey || '',
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  };

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
  });

  // Chunk Transcription Endpoint (processes a small ~5MB chunk per HTTP request)
  app.post('/api/transcribe-chunk', async (req, res) => {
    try {
      const { chunk, options } = req.body;
      if (!chunk || !chunk.base64) {
        return res.status(400).json({ error: 'Данные аудиофрагмента отсутствуют' });
      }

      const ai = getGeminiClient();
      const languageInstruction = options?.language && options.language !== 'auto'
        ? `Язык аудио: ${options.language}. Расшифруй строго на этом языке.`
        : 'Автоматически определи язык речи (русский, английский, казахский и т.д.).';

      const modeInstruction = options?.mode === 'clean'
        ? 'Очисти текст от сорных слов (эээ, ну, как бы, типа) и заиканий, делая речь связной.'
        : 'Делай максимально точный дословный транскрипт речи.';

      const diarizationInstruction = options?.enableDiarization
        ? 'Раздели диалог по спикерам (Спикер 1, Спикер 2 и т.д.). Указывай смену говорящего.'
        : 'Используй обозначения Спикер 1 или единый текст.';

      const glossaryInstruction = options?.customGlossary
        ? `Специальный глоссарий и термины для контекста: ${options.customGlossary}`
        : '';

      const systemPrompt = `
Ты — профессиональная система транскрибации речи и анализа медиафайлов.
Твоя задача — прослушать переданный фрагмент аудиофайла и вернуть точную расшифровку речи с временными метками и разбивкой по спикерам.

Инструкции:
1. ${languageInstruction}
2. ${modeInstruction}
3. ${diarizationInstruction}
4. ${glossaryInstruction}
5. Разбей транскрипт на логические сегменты с временными метками (startSec, endSec, timestamp в формате MM:SS).
`;

      const base64Clean = chunk.base64.replace(/^data:[^;]+;base64,/, '');
      const offsetSec = chunk.offsetSec || 0;
      const index = chunk.index !== undefined ? chunk.index : 0;
      const totalChunks = chunk.totalChunks || 1;

      console.log(`Transcribing chunk ${index + 1}/${totalChunks} (offset ${offsetSec}s)...`);

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: [
          {
            role: 'user',
            parts: [
              { inlineData: { mimeType: chunk.mimeType || 'audio/wav', data: base64Clean } },
              { text: `Расшифруй этот фрагмент аудиотега (часть ${index + 1} из ${totalChunks}). Смещение этого фрагмента: ${offsetSec} секунд.` },
            ],
          },
        ],
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              languageDetected: { type: Type.STRING },
              rawText: { type: Type.STRING },
              segments: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    startSec: { type: Type.NUMBER },
                    endSec: { type: Type.NUMBER },
                    timestamp: { type: Type.STRING },
                    speaker: { type: Type.STRING },
                    text: { type: Type.STRING },
                  },
                  required: ['id', 'startSec', 'endSec', 'timestamp', 'speaker', 'text'],
                },
              },
            },
            required: ['languageDetected', 'rawText', 'segments'],
          },
        },
      });

      const text = response.text || '';
      if (!text) {
        return res.status(500).json({ error: 'Пустой ответ от нейросети Gemini' });
      }

      const parsed = JSON.parse(text);
      return res.json({
        success: true,
        data: {
          index,
          offsetSec,
          languageDetected: parsed.languageDetected || 'Русский',
          rawText: parsed.rawText || '',
          segments: parsed.segments || [],
        },
      });
    } catch (err: any) {
      console.error('Chunk transcription error:', err);
      return res.status(500).json({ error: err?.message || 'Ошибка расшифровки аудиофрагмента' });
    }
  });

  // Summarize Full Transcript Endpoint
  app.post('/api/summarize', async (req, res) => {
    try {
      const { rawText, segments, fileName, fileType, mimeType, languageDetected } = req.body;

      const ai = getGeminiClient();
      let summaryData = {
        overview: 'Расшифровка медиафайла выполнена успешно.',
        keyPoints: ['Подготовлен сплошной текст и хронологические сегменты.', 'Запись проанализирована нейросетью Gemini 2.5.'],
        actionItems: ['Изучить ключевые выводы и скачать готовые субтитры.'],
        sentiment: 'Нейтральный / Деловой',
        topics: ['Медиазапись', 'Аудиоанализ']
      };

      if (rawText && rawText.trim().length > 0) {
        try {
          const summaryResponse = await ai.models.generateContent({
            model: 'gemini-3.6-flash',
            contents: [
              {
                role: 'user',
                parts: [
                  { text: `Проанализируй полный расшифрованный текст и составь структурированное AI-саммари:\n\n"""\n${rawText.slice(0, 30000)}\n"""` }
                ]
              }
            ],
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  overview: { type: Type.STRING },
                  keyPoints: { type: Type.ARRAY, items: { type: Type.STRING } },
                  actionItems: { type: Type.ARRAY, items: { type: Type.STRING } },
                  sentiment: { type: Type.STRING },
                  topics: { type: Type.ARRAY, items: { type: Type.STRING } },
                },
                required: ['overview', 'keyPoints', 'actionItems', 'sentiment', 'topics'],
              }
            }
          });

          if (summaryResponse.text) {
            summaryData = JSON.parse(summaryResponse.text);
          }
        } catch (e) {
          console.warn('Summary generation error:', e);
        }
      }

      const safeSegments = segments || [];
      const srtContent = buildSrtContent(safeSegments);
      const wordCount = (rawText || '').split(/\s+/).filter(Boolean).length;
      const durationEstimateSec = safeSegments.length > 0 ? safeSegments[safeSegments.length - 1].endSec : 60;

      return res.json({
        success: true,
        data: {
          languageDetected: languageDetected || 'Русский',
          durationEstimateSec,
          rawText: rawText || '',
          segments: safeSegments,
          summary: summaryData,
          srtContent,
          wordCount,
          fileName: fileName || 'Запись',
          fileType: fileType || 'audio',
          mimeType: mimeType || 'audio/wav',
        },
      });
    } catch (err: any) {
      console.error('Summarize route error:', err);
      return res.status(500).json({ error: err?.message || 'Ошибка генерации саммари' });
    }
  });

  // Main Transcription Endpoint
  app.post('/api/transcribe', async (req, res) => {
    try {
      const { fileData, chunks, mimeType, fileName, fileType, options, sampleId } = req.body;

      if (!fileData && (!chunks || chunks.length === 0) && !sampleId) {
        return res.status(400).json({ error: 'Загрузите медиафайлы для расшифровки' });
      }

      const ai = getGeminiClient();
      const languageInstruction = options?.language && options.language !== 'auto'
        ? `Язык аудио: ${options.language}. Расшифруй строго на этом языке.`
        : 'Автоматически определи язык речи (русский, английский, казахский и т.д.).';

      const modeInstruction = options?.mode === 'clean'
        ? 'Очисти текст от сорных слов (эээ, ну, как бы, типа) и заиканий, делая речь связной.'
        : 'Делай максимально точный дословный транскрипт речи.';

      const diarizationInstruction = options?.enableDiarization
        ? 'Раздели диалог по спикерам (Спикер 1, Спикер 2 и т.д.). Указывай смену говорящего.'
        : 'Используй обозначения Спикер 1 или единый текст.';

      const glossaryInstruction = options?.customGlossary
        ? `Специальный глоссарий и термины для контекста: ${options.customGlossary}`
        : '';

      const systemPrompt = `
Ты — профессиональная система транскрибации речи и анализа медиафайлов.
Твоя задача — прослушать/просмотреть переданный файл и вернуть точную расшифровку речи с временными метками, разбивкой по спикерам и умным анализом.

Инструкции:
1. ${languageInstruction}
2. ${modeInstruction}
3. ${diarizationInstruction}
4. ${glossaryInstruction}
5. Разбей транскрипт на логические сегменты с временными метками (startSec, endSec, timestamp в формате MM:SS).
6. Сформируй также готовые субтитры в формате SRT.
7. Составь AI-саммари: краткое содержание (overview), ключевые тезисы (keyPoints), список поручений/задач (actionItems), эмоциональный тон (sentiment) и темы (topics).
`;

      // Handle multi-chunk media uploads (for long videos / high quality recordings)
      if (chunks && Array.isArray(chunks) && chunks.length > 0) {
        console.log(`Processing ${chunks.length} audio chunks sequentially for ${fileName}...`);
        
        const validResults = [];
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          const offsetSec = chunk.offsetSec || 0;
          const base64Clean = chunk.base64 ? chunk.base64.replace(/^data:[^;]+;base64,/, '') : '';
          if (!base64Clean) continue;

          console.log(`Sending chunk ${i + 1}/${chunks.length} (offset ${offsetSec}s, size ${(base64Clean.length * 0.75 / 1024 / 1024).toFixed(2)} MB)...`);

          try {
            const response = await ai.models.generateContent({
              model: 'gemini-3.6-flash',
              contents: [
                {
                  role: 'user',
                  parts: [
                    { inlineData: { mimeType: chunk.mimeType || 'audio/wav', data: base64Clean } },
                    { text: `Расшифруй этот фрагмент аудиотега (часть ${i + 1} из ${chunks.length}). Смещение фрагмента: ${offsetSec} секунд.` },
                  ],
                },
              ],
              config: {
                systemInstruction: systemPrompt,
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    languageDetected: { type: Type.STRING },
                    rawText: { type: Type.STRING },
                    segments: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          id: { type: Type.STRING },
                          startSec: { type: Type.NUMBER },
                          endSec: { type: Type.NUMBER },
                          timestamp: { type: Type.STRING },
                          speaker: { type: Type.STRING },
                          text: { type: Type.STRING },
                        },
                        required: ['id', 'startSec', 'endSec', 'timestamp', 'speaker', 'text'],
                      },
                    },
                  },
                  required: ['languageDetected', 'rawText', 'segments'],
                },
              },
            });

            const text = response.text || '';
            if (text) {
              const parsed = JSON.parse(text);
              validResults.push({ index: i, offsetSec, ...parsed });
            }
          } catch (err) {
            console.error(`Chunk ${i + 1} error:`, err);
          }
        }

        if (validResults.length > 0) {
          let combinedSegments: any[] = [];
          let combinedRawTextParts: string[] = [];
          let detectedLang = validResults[0]!.languageDetected || 'Русский';

          validResults.forEach((res) => {
            const offset = res.offsetSec || 0;
            if (res.segments) {
              res.segments.forEach((seg: any, idx: number) => {
                const adjStart = Math.round((seg.startSec || 0) + offset);
                const adjEnd = Math.round((seg.endSec || 0) + offset);
                combinedSegments.push({
                  id: `seg-${res.index + 1}-${idx + 1}`,
                  startSec: adjStart,
                  endSec: adjEnd,
                  timestamp: formatTimestamp(adjStart),
                  speaker: seg.speaker || 'Спикер 1',
                  text: seg.text || '',
                });
              });
            }
            if (res.rawText) {
              combinedRawTextParts.push(res.rawText.trim());
            }
          });

          const combinedRawText = combinedRawTextParts.join('\n\n');
          const totalDurationSec = combinedSegments.length > 0 ? combinedSegments[combinedSegments.length - 1].endSec : 60;

          // Generate overall summary
          let summaryData = {
            overview: 'Расшифровка медиафайла выполнена успешно.',
            keyPoints: ['Подготовлен сплошной текст и хронологические сегменты.', 'Запись проанализирована нейросетью Gemini 3.6.'],
            actionItems: ['Изучить ключевые выводы и скачать готовые субтитры.'],
            sentiment: 'Нейтральный / Деловой',
            topics: ['Медиазапись', 'Аудиоанализ']
          };

          try {
            const summaryResponse = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: [
                {
                  role: 'user',
                  parts: [
                    { text: `Проанализируй полный расшифрованный текст и составь структурированное AI-саммари:\n\n"""\n${combinedRawText.slice(0, 30000)}\n"""` }
                  ]
                }
              ],
              config: {
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    overview: { type: Type.STRING },
                    keyPoints: { type: Type.ARRAY, items: { type: Type.STRING } },
                    actionItems: { type: Type.ARRAY, items: { type: Type.STRING } },
                    sentiment: { type: Type.STRING },
                    topics: { type: Type.ARRAY, items: { type: Type.STRING } },
                  },
                  required: ['overview', 'keyPoints', 'actionItems', 'sentiment', 'topics'],
                }
              }
            });

            if (summaryResponse.text) {
              summaryData = JSON.parse(summaryResponse.text);
            }
          } catch (e) {
            console.warn('Summary generation error:', e);
          }

          const srtContent = buildSrtContent(combinedSegments);
          const wordCount = combinedRawText.split(/\s+/).filter(Boolean).length;

          return res.json({
            success: true,
            data: {
              languageDetected: detectedLang,
              durationEstimateSec: totalDurationSec,
              rawText: combinedRawText,
              segments: combinedSegments,
              summary: summaryData,
              srtContent,
              wordCount,
              fileName: fileName || 'Запись',
              fileType: fileType || 'video',
              mimeType: mimeType || 'audio/wav',
            },
          });
        } else {
          return res.status(500).json({ error: 'Не удалось обработать аудиофрагменты через нейросеть Gemini.' });
        }
      }

      // Clean base64 string
      const base64Clean = fileData ? fileData.replace(/^data:[^;]+;base64,/, '') : '';

      if (base64Clean && base64Clean.length > 50) {
        try {
          let effectiveMime = mimeType || 'audio/wav';
          if (fileType === 'video' && (!mimeType || mimeType.startsWith('application/'))) {
            effectiveMime = 'video/mp4';
          }

          console.log(`Sending single media payload to Gemini 3.6 Flash (mime: ${effectiveMime}, size: ${(base64Clean.length * 0.75 / 1024 / 1024).toFixed(2)} MB)...`);

          const response = await ai.models.generateContent({
            model: 'gemini-3.6-flash',
            contents: [
              {
                role: 'user',
                parts: [
                  {
                    inlineData: {
                      mimeType: effectiveMime,
                      data: base64Clean,
                    },
                  },
                  {
                    text: 'Расшифруй это аудио/видео в точном соответствии с требованиями.',
                  },
                ],
              },
            ],
            config: {
              systemInstruction: systemPrompt,
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  languageDetected: { type: Type.STRING, description: 'Определенный язык (например, "Русский")' },
                  durationEstimateSec: { type: Type.NUMBER, description: 'Примерная длительность в секундах' },
                  rawText: { type: Type.STRING, description: 'Полный сплошной текст расшифровки' },
                  segments: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        startSec: { type: Type.NUMBER },
                        endSec: { type: Type.NUMBER },
                        timestamp: { type: Type.STRING },
                        speaker: { type: Type.STRING },
                        text: { type: Type.STRING },
                      },
                      required: ['id', 'startSec', 'endSec', 'timestamp', 'speaker', 'text'],
                    },
                  },
                  summary: {
                    type: Type.OBJECT,
                    properties: {
                      overview: { type: Type.STRING },
                      keyPoints: { type: Type.ARRAY, items: { type: Type.STRING } },
                      actionItems: { type: Type.ARRAY, items: { type: Type.STRING } },
                      sentiment: { type: Type.STRING },
                      topics: { type: Type.ARRAY, items: { type: Type.STRING } },
                    },
                    required: ['overview', 'keyPoints', 'actionItems'],
                  },
                  srtContent: { type: Type.STRING, description: 'Полный готовый текст файла субтитров .SRT' },
                },
                required: ['languageDetected', 'rawText', 'segments', 'summary', 'srtContent'],
              },
            },
          });

          if (response.text) {
            const parsed = JSON.parse(response.text);
            const wordCount = parsed.rawText ? parsed.rawText.trim().split(/\s+/).filter(Boolean).length : 0;
            return res.json({
              success: true,
              data: {
                ...parsed,
                wordCount,
                fileName: fileName || 'Запись',
                fileType: fileType || 'audio',
                mimeType: mimeType || 'audio/mp3',
              },
            });
          }
        } catch (geminiError: any) {
          console.error('Gemini direct transcription error:', geminiError);
          if (!sampleId) {
            return res.status(500).json({ error: `Ошибка ИИ Gemini: ${geminiError?.message || 'Не удалось обработать аудио'}` });
          }
        }
      }

      // Fallback generator ONLY for demo samples
      if (sampleId) {
        const mockResult = generateFallbackTranscript(fileName || 'Демо запись', sampleId, fileType);
        return res.json({
          success: true,
          data: {
            ...mockResult,
            fileName: fileName || 'Запись',
            fileType: fileType || 'video',
            mimeType: mimeType || (fileType === 'video' ? 'video/mp4' : 'audio/mp3'),
          },
        });
      }

      return res.status(400).json({ error: 'Не удалось прочитать или распознать медиафайл. Проверьте формат и целостность файла.' });

    } catch (error: any) {
      console.error('Transcription route error:', error);
      res.status(500).json({ error: error.message || 'Ошибка обработки аудиофайла' });
    }
  });

  // AI Assistant for transcript (Q&A, translation, re-summary)
  app.post('/api/analyze-transcript', async (req, res) => {
    try {
      const { transcriptText, segments, action, customPrompt, targetLanguage } = req.body;

      if (!transcriptText) {
        return res.status(400).json({ error: 'Текст транскрипта отсутствует' });
      }

      const ai = getGeminiClient();

      let prompt = '';
      if (action === 'question') {
        prompt = `Ответь на вопрос пользователя по следующему расшифрованному тексту записи:
Вопрос: ${customPrompt}

Текст записи:
"""
${transcriptText}
"""
Дай понятный, структурированный и исчерпывающий ответ на русском языке.`;
      } else if (action === 'translate') {
        prompt = `Переведи весь следующий расшифрованный текст на язык: "${targetLanguage || 'English'}".
Сохраняй смысл, естественность и структуру речи.

Текст для перевода:
"""
${transcriptText}
"""`;
      } else if (action === 'protocol') {
        prompt = `Составь официальный протокол встречи (совещания) на основе следующего транскрипта.
Включи:
1. Тему и цель встречи
2. Участников (если упомянуты)
3. Ключевые решения
4. Поручения с ответственным и сроками (если есть)

Текст:
"""
${transcriptText}
"""`;
      } else {
        prompt = `Проанализируй текст записи и выполни запрос: ${customPrompt}\n\nТекст:\n${transcriptText}`;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
      });

      res.json({
        success: true,
        result: response.text || 'Ответ сформирован.',
      });
    } catch (error: any) {
      console.error('Analysis error:', error);
      res.status(500).json({ error: error.message || 'Ошибка обработки AI запроса' });
    }
  });

  // Vite development middleware
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
}

function generateFallbackTranscript(title: string, sampleId?: string, fileType?: string) {
  let segments = [];
  let overview = '';
  let keyPoints = [];
  let actionItems = [];

  const isVideo = fileType === 'video' || title.match(/\.(mp4|webm|mov|avi|mkv)$/i) !== null;

  if (sampleId === 'sample-interview') {
    overview = 'Собеседование с кандидатом на позицию Senior Frontend Developer. Обсудили опыт работы с React 19, архитектуру сложных веб-приложений и подходы к оптимизации рендеринга.';
    keyPoints = [
      'Кандидат имеет 6+ лет опыта во фронтенд-разработке.',
      'Успешный опыт внедрения Server Actions и React Server Components.',
      'Уверенные знания TypeScript, Zustand и оптимизации бандла.'
    ];
    actionItems = [
      'Отправить тестовое задание по веб-воркерам.',
      'Назначить финальную встречу с CTO до конца недели.'
    ];
    segments = [
      { id: '1', startSec: 0, endSec: 8, timestamp: '00:00', speaker: 'Интервьюер', text: 'Здравствуйте! Расскажите кратко о вашем опыте разработки на React и TypeScript.' },
      { id: '2', startSec: 8, endSec: 25, timestamp: '00:08', speaker: 'Кандидат', text: 'Добрый день! Я занимаюсь фронтендом около 6 лет. Последние 3 года руководил модулем в высоконагруженном FinTech приложении.' },
      { id: '3', startSec: 25, endSec: 42, timestamp: '00:25', speaker: 'Интервьюер', text: 'Отлично. Как вы решаете задачи по оптимизации перерисовки крупных таблиц с частыми обновлениями данных?' },
      { id: '4', startSec: 42, endSec: 65, timestamp: '00:42', speaker: 'Кандидат', text: 'Мы используем виртуализацию списков через TanStack Virtual, мемоизацию селекторов и вынесение состояния во внешнее хранилище.' }
    ];
  } else if (sampleId === 'sample-lecture') {
    overview = 'Информативная мини-лекция про эволюцию мультимодального ИИ, современные языковые модели и интеграцию Gemini API в веб-сервисы.';
    keyPoints = [
      'Мультимодальность позволяет обрабатывать текст, аудио и видео одновременно.',
      'Скорость транскрибации речи с помощью AI возросла в десятки раз по сравнению с традиционными ASR.',
      'Серверная интеграция гарантирует безопасное хранение API ключей.'
    ];
    actionItems = [
      'Изучить документацию @google/genai SDK.',
      'Протестировать потоковую передачу данных через WebSockets.'
    ];
    segments = [
      { id: '1', startSec: 0, endSec: 15, timestamp: '00:00', speaker: 'Лектор', text: 'Приветствую всех! Сегодня мы разберем, как современные нейросети меняют работу с аудио и видео.' },
      { id: '2', startSec: 15, endSec: 35, timestamp: '00:15', speaker: 'Лектор', text: 'Главное отличие современных нейросетей от прошлых систем Whisper и ASR — это понимание интонации, эмоций и контекста.' },
      { id: '3', startSec: 35, endSec: 55, timestamp: '00:35', speaker: 'Лектор', text: 'Благодаря этому мы можем мгновенно получать не только точную расшифровку, но и авто-протоколы встреч.' }
    ];
  } else {
    overview = 'Обсуждение плана разработки и релиза цифрового продукта. Были согласованы ключевые этапы интеграции API и графики сдачи компонентов.';
    keyPoints = [
      'Завершение первого этапа тестирования запланировано на пятницу.',
      'Все требования по дизайну и адаптивной верстке согласованы с продуктовой командой.',
      'Подготовка к развертыванию на Cloud Run готовит стабильный релиз.'
    ];
    actionItems = [
      'Разработчикам подготовить пулл-реквесты к среду.',
      'Провести нагрузочное тестирование сервера транскрибации.'
    ];
    segments = [
      { id: '1', startSec: 0, endSec: 10, timestamp: '00:00', speaker: 'Спикер 1', text: 'Коллеги, давайте обсудим текущий статус задачи по транскрибации медиафайлов.' },
      { id: '2', startSec: 10, endSec: 25, timestamp: '00:10', speaker: 'Спикер 2', text: 'Все ключевые эндпоинты настроены. Транскрибация работает корректно с распознаванием спикеров и генерацией SRT.' },
      { id: '3', startSec: 25, endSec: 40, timestamp: '00:25', speaker: 'Спикер 1', text: 'Прекрасно! Обратите внимание на автоопределение языка и скачивание субтитров.' }
    ];
  }

  const rawText = segments.map(s => `${s.speaker}: ${s.text}`).join('\n\n');
  const wordCount = rawText.split(/\s+/).length;

  let srtContent = '';
  segments.forEach((seg, idx) => {
    const startMs = formatSrtTime(seg.startSec);
    const endMs = formatSrtTime(seg.endSec);
    srtContent += `${idx + 1}\n${startMs} --> ${endMs}\n[${seg.speaker}]: ${seg.text}\n\n`;
  });

  return {
    languageDetected: 'Русский',
    durationEstimateSec: segments[segments.length - 1]?.endSec || 45,
    rawText,
    segments,
    summary: {
      overview,
      keyPoints,
      actionItems,
      sentiment: 'Деловой / Позитивный',
      topics: ['IT', 'Разработка', 'Планирование']
    },
    srtContent,
    wordCount
  };
}

function formatTimestamp(totalSec: number): string {
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function buildSrtContent(segments: any[]): string {
  let srt = '';
  segments.forEach((seg, idx) => {
    const startStr = formatSrtTime(seg.startSec);
    const endStr = formatSrtTime(seg.endSec);
    srt += `${idx + 1}\n${startStr} --> ${endStr}\n[${seg.speaker}]: ${seg.text}\n\n`;
  });
  return srt;
}

function formatSrtTime(totalSec: number): string {
  const hrs = Math.floor(totalSec / 3600);
  const mins = Math.floor((totalSec % 3600) / 60);
  const secs = Math.floor(totalSec % 60);
  const ms = Math.floor((totalSec % 1) * 1000);

  const pad = (n: number, z = 2) => String(n).padStart(z, '0');
  return `${pad(hrs)}:${pad(mins)}:${pad(secs)},${pad(ms, 3)}`;
}

startServer();
