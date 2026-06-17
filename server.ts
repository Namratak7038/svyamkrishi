/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "10mb" }));

// Lazy initializer for Gemini Client
let aiInstance: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiInstance) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY environment variable is not defined in Secrets.");
    }
    aiInstance = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiInstance;
}

// Helper to sanitize JSON response from Gemini
function cleanJsonMarkdown(text: string): string {
  let cleaned = text.trim();
  if (cleaned.startsWith("```json")) {
    cleaned = cleaned.substring(7);
  } else if (cleaned.startsWith("```")) {
    cleaned = cleaned.substring(3);
  }
  if (cleaned.endsWith("```")) {
    cleaned = cleaned.substring(0, cleaned.length - 3);
  }
  return cleaned.trim();
}

// 1. AI Crop Recommendation API
app.post("/api/gemini/crop-recommendation", async (req, res) => {
  try {
    const { soilType, landArea, waterAvailability, rainfall, season, district, state, preferredLang } = req.body;
    const ai = getGeminiClient();

    const prompt = `
      You are an expert agronomist advisor for Indian farmers.
      Analyze the following options and provide agricultural recommendation:
      - Sowing District/State: ${district}, ${state}
      - Soil Type: ${soilType}
      - Farm Area: ${landArea} Acres
      - Water Availability: ${waterAvailability}
      - Annual/Seasonal Rainfall estimate: ${rainfall} mm
      - Sowing Season: ${season}

      Return a response in strict JSON format. Translate all text outputs into ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      JSON Schema required:
      {
        "recommendedCrops": [
          {
            "cropName": "Name of crop",
            "suitabilityScore": 95 (percentage integer),
            "expectedYield": "Estimated range (e.g., 12-15 Quintals/Acre)",
            "expectedProfit": "Estimated net profit per acre (e.g., ₹45,000)",
            "riskLevel": "Low" | "Medium" | "High",
            "riskReason": "Short phrase describing the main threat",
            "reasoning": "1-2 brief sentences explanation why this is suitable for this soil and season"
          }
        ],
        "soilHealthAdvice": "Short tactical soil prep advise for the farmer",
        "irrigationSuggestion": "Short action item on irrigation frequency"
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    const jsonText = cleanJsonMarkdown(response.text || "{}");
    res.json(JSON.parse(jsonText));
  } catch (error: any) {
    console.error("Crop Recommendation Error:", error);
    res.status(500).json({ error: error.message || "Failed to generate recommendation" });
  }
});

// 2. Weather Advisory Suggestions API
app.post("/api/gemini/weather-advisory", async (req, res) => {
  try {
    const { state, district, crops, preferredLang } = req.body;
    const ai = getGeminiClient();

    const prompt = `
      You are SvyamKrishi AI weather agro-intelligence service.
      Generate 3 highly practical weather advisory suggestions based on crops: ${crops?.join(", ") || "suggested crops"} in ${district}, ${state}.
      The advice should sound natural and human, tailored specifically to farmers.
      Example: "Heavy dew is expected over Satara tonight. Apply copper fungicide to your tomatoes."

      Return a response in strict JSON format. Language of reply: ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      JSON Schema required:
      {
        "bulletPoints": [
          "Suggestion 1",
          "Suggestion 2",
          "Suggestion 3"
        ],
        "warningAlert": "A short weather risk alert (e.g. alert for unseasonal storm or heatwave, or 'None')"
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    const jsonText = cleanJsonMarkdown(response.text || "{}");
    res.json(JSON.parse(jsonText));
  } catch (error: any) {
    console.error("Weather Advisory Error:", error);
    res.status(500).json({ error: error.message || "Failed to generate weather advisory" });
  }
});

// 3. AI Yield Prediction API
app.post("/api/gemini/yield-prediction", async (req, res) => {
  try {
    const { cropType, landArea, soilType, fertilizerUsage, rainfall, preferredLang } = req.body;
    const ai = getGeminiClient();

    const prompt = `
      You are an agricultural yield forecasting engine utilizing Random Forest & XGBoost modeling variables.
      Analyze options to forecast yield, revenue, and profit metrics:
      - Crop: ${cropType}
      - Land Area: ${landArea} Acres
      - Soil Type: ${soilType}
      - Fertilizer Applied (N-P-K context): ${fertilizerUsage}
      - Rainfall projection: ${rainfall}
      
      Provide expected numeric outcome projections.
      Return a response in strict JSON format. Translate readable sentences into ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      JSON Schema required:
      {
        "expectedYieldQuintals": 45 (numeric integer value representing total yield for entire land area),
        "confidenceScore": 92 (percentage integer),
        "expectedRevenueINR": 112500 (number),
        "expectedNetProfitINR": 76000 (number),
        "yieldDrivers": [
          {"factor": "Soil suitability", "impact": "High positive"},
          {"factor": "Rainfall adequacy", "impact": "Moderate"}
        ],
        "soilNutritionOptimization": "One short optimization sentence"
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    res.json(JSON.parse(cleanJsonMarkdown(response.text || "{}")));
  } catch (error: any) {
    console.error("Yield Prediction Error:", error);
    res.status(500).json({ error: error.message || "Failed to predict yield" });
  }
});

// 4. Crop Disease Diagnostic Scanner API
app.post("/api/gemini/crop-disease-scan", async (req, res) => {
  try {
    const { base64Image, diagnosticPresetName, cropContext, preferredLang } = req.body;
    const ai = getGeminiClient();

    let geminiResponse;
    const analysisInstructions = `
      You are SvyamKrishi AI Crop leaf pathologist diagnostics engine.
      Identify leaf disease details for: ${cropContext || "specified cop"}.
      Diagnostic reference target: ${diagnosticPresetName || "unknown pathology"}.
      
      Response MUST be in strict JSON format. Translate all textual solutions into ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      
      JSON Schema structure:
      {
        "diseaseName": "Recognizable Name with biological nomenclature",
        "confidenceScore": 89 (percentage integer),
        "severityLevel": "Low" | "Medium" | "High",
        "description": "Short explanation of how this disease manifests",
        "treatmentChemical": "Name exact Indian pesticide, fungicide, or chemical wash dosage to use immediately",
        "treatmentOrganic": "Provide homemade natural spray or organic mulch solutions",
        "preventionMethods": [
          "Crop rotation advice",
          "Water splash protection directive",
          "Balanced leaf aeration"
        ]
      }
    `;

    if (base64Image) {
      // Image diagnostic query (using parts array with image)
      const imagePart = {
        inlineData: {
          mimeType: "image/jpeg",
          data: base64Image
        }
      };
      const textPart = {
        text: `Analyze the attached leaf crop image. Instructions: ${analysisInstructions}`
      };

      geminiResponse = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: { parts: [imagePart, textPart] },
        config: {
          responseMimeType: "application/json"
        }
      });
    } else {
      // Text-based preset simulation path
      geminiResponse = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: `Simulate leaf diagnostics for leaf with symptoms: "${diagnosticPresetName}". Instructions: ${analysisInstructions}`,
        config: {
          responseMimeType: "application/json"
        }
      });
    }

    res.json(JSON.parse(cleanJsonMarkdown(geminiResponse.text || "{}")));
  } catch (error: any) {
    console.error("Disease Diagnostic Error:", error);
    res.status(500).json({ error: error.message || "pathology identification failed" });
  }
});

// 5. AI Voice Assistant Support Chat API
app.post("/api/gemini/voice-assistant-chat", async (req, res) => {
  try {
    const { userMessage, farmerState, preferredLang } = req.body;
    const ai = getGeminiClient();

    const systemPrompt = `
      You are SvyamKrishi, a warm, highly compassionate AI voice companion or "Sahayak" for Indian farmers.
      You help with sowing guidelines, mandi prices, unseasonal rain preparations, loan interest queries, or general support.
      The user acts as a farmer from ${farmerState || "rural India"}.
      Keep your response brief (maximum 2-3 short sentences), conversational, highly encouraging, and strictly practical.
      
      Return a response in strict JSON format. The language of response and audioScript MUST be: ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      
      JSON Schema schema:
      {
        "replyText": "Rich, loving response text for printing inside chat bubble.",
        "audioScript": "Clean spoken-friendly script without asterisks or bullet points, perfect for text-to-speech synthesis aloud."
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: `Farmer asks/speaks: "${userMessage}"`,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json"
      }
    });

    res.json(JSON.parse(cleanJsonMarkdown(response.text || "{}")));
  } catch (error: any) {
    console.error("Voice Assistant Error:", error);
    res.status(500).json({ error: error.message || "Failed to compile audio discussion" });
  }
});

// 6. Market Price Advisory Recommendation API
app.post("/api/gemini/market-advisory", async (req, res) => {
  try {
    const { cropName, currentPrice, preferredLang } = req.body;
    const ai = getGeminiClient();

    const prompt = `
      You are an expert Mandi economic advisor for Indian APMCs.
      Analyze market variables for:
      - Crop: ${cropName}
      - Current mandi rate: ₹${currentPrice} per Quintal

      Recommend if the farmer should sell now, hold, or ship to a different district APMC.
      Return a response in strict JSON format. Translate text into ${preferredLang === "hi" ? "Hindi (हिन्दी)" : preferredLang === "mr" ? "Marathi (मराठी)" : "English"}.
      JSON Schema required:
      {
        "statusAction": "SELL NOW" | "HOLD CROP" | "DIVERSIFY MANDI",
        "priceTrendNextWeek": "Estimate (e.g. Expected increase of ₹150-200)",
        "advisoryVerdict": "Detailed 1-2 sentence economic justification",
        "bestMarketMandi": "Mandi Name suggestions with rates"
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    res.json(JSON.parse(cleanJsonMarkdown(response.text || "{}")));
  } catch (error: any) {
    console.error("Market Advisory Error:", error);
    res.status(500).json({ error: error.message || "failed to consult APMC market trends" });
  }
});


// Serves the client-side SPA or Vite Dev Middleware
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[SvyamKrishi Server] running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
