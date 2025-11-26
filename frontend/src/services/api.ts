import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for agent processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for API responses
export interface PlaceResult {
  name: string;
  rating?: number;
  address?: string;
  coordinates?: [number, number]; // [lng, lat]
  distance_km?: number;
  description?: string;
  price?: string;
  reviews?: number;
  hours?: string;
  type?: string;
  phone?: string;
  website?: string;
}

export interface SearchResponse {
  success: boolean;
  response?: string;
  places: PlaceResult[];
  error?: string;
  query: string;
  processing_time?: number;
}

export interface HealthResponse {
  status: string;
  agent_status: string;
  version: string;
}

export interface SearchRequest {
  query: string;
  location?: string;
}

// API functions
export const searchPlaces = async (request: SearchRequest): Promise<SearchResponse> => {
  try {
    console.log('Making search request:', request);
    const response = await api.post<SearchResponse>('/search', request);
    console.log('Search response received:', response.data);
    return response.data;
  } catch (error) {
    console.error('Search error:', error);
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // Server responded with error status
        console.error('Server error response:', error.response.data);
        throw new Error(`Search failed: ${error.response.data?.error || error.response.statusText}`);
      } else if (error.request) {
        // Request was made but no response received (likely timeout)
        console.error('No response received, possibly timeout. Request:', error.request);
        throw new Error('Request timeout. The search is taking longer than expected. Please try again.');
      } else {
        // Something else happened
        console.error('Request setup error:', error.message);
        throw new Error(`Request error: ${error.message}`);
      }
    } else {
      console.error('Unexpected error:', error);
      throw new Error('An unexpected error occurred during search');
    }
  }
};

export const checkHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Health check failed: ${error.message}`);
    }
    throw new Error('Failed to check backend health');
  }
};

export const getConversationHistory = async () => {
  try {
    const response = await api.get('/conversation/history');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to get conversation history: ${error.message}`);
    }
    throw new Error('Failed to get conversation history');
  }
};

export const resetAgent = async () => {
  try {
    const response = await api.post('/agent/reset');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to reset agent: ${error.message}`);
    }
    throw new Error('Failed to reset agent');
  }
};

export const getAvailableModels = async () => {
  try {
    const response = await api.get('/models/available');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to get available models: ${error.message}`);
    }
    throw new Error('Failed to get available models');
  }
};

// Memory management functions
export interface MemoryInfo {
  type: string;
  description: string;
  message_count: number;
}

export interface MemoryType {
  type: string;
  name: string;
  description: string;
  pros: string[];
  cons: string[];
}

export const getMemoryInfo = async (): Promise<MemoryInfo> => {
  try {
    const response = await api.get<MemoryInfo>('/agent/memory/info');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to get memory info: ${error.message}`);
    }
    throw new Error('Failed to get memory info');
  }
};

export const getMemoryTypes = async (): Promise<{ memory_types: MemoryType[] }> => {
  try {
    const response = await api.get<{ memory_types: MemoryType[] }>('/agent/memory/types');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to get memory types: ${error.message}`);
    }
    throw new Error('Failed to get memory types');
  }
};

export const switchMemoryType = async (memoryType: string) => {
  try {
    const response = await api.post(`/agent/memory/switch?memory_type=${memoryType}`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to switch memory type: ${error.message}`);
    }
    throw new Error('Failed to switch memory type');
  }
};