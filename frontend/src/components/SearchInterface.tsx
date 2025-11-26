'use client';

import React, { useState, useCallback } from 'react';
import {
  Paper,
  TextField,
  Button,
  Box,
  Typography,
  CircularProgress,
  IconButton,
  Tooltip,
  LinearProgress,
  Divider,
  Chip,
  Card,
  CardContent
} from '@mui/material';
import {
  Search as SearchIcon,
  LocationOn as LocationIcon,
  Clear as ClearIcon,
  AccessTime as TimeIcon
} from '@mui/icons-material';
import { searchPlaces, PlaceResult, SearchResponse } from '@/services/api';

interface SearchInterfaceProps {
  onResults: (results: PlaceResult[], query: string) => void;
  onError: (error: string) => void;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ onResults, onError }) => {
  const [query, setQuery] = useState('');
  const [location, setLocation] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastResponse, setLastResponse] = useState<SearchResponse | null>(null);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  // Travel-focused search suggestions
  const suggestions = [
    'Tourist attractions near Eiffel Tower',
    'Hotels walking distance from Colosseum',
    'Best viewpoints in Santorini',
    'Local markets in Bangkok',
    'Museums near Central Park',
    'Restaurants with city views',
    'Airport shuttle services',
    'Currency exchange near me',
    'Nightlife in Barcelona',
    'Traditional cafes in Vienna',
    'Shopping districts in Tokyo',
    'Public transport hubs'
  ];

  const locationSuggestions = [
    'Paris, France',
    'San Francisco, CA',
    'New York, NY',
    'London, UK',
    'Tokyo, Japan',
    'Barcelona, Spain'
  ];

  const handleSearch = useCallback(async () => {
    if (!query.trim()) {
      onError('Please enter a search query');
      return;
    }

    setIsLoading(true);
    setLastResponse(null);

    try {
      const searchRequest = {
        query: query.trim(),
        location: location.trim() || undefined
      };

      const response = await searchPlaces(searchRequest);
      setLastResponse(response);

      if (response.success) {
        onResults(response.places, response.query);
        
        // Add to search history
        const fullQuery = location ? `${query} in ${location}` : query;
        setSearchHistory(prev => {
          const newHistory = [fullQuery, ...prev.filter(h => h !== fullQuery)].slice(0, 5);
          return newHistory;
        });
      } else {
        onError(response.error || 'Search failed');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [query, location, onResults, onError]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isLoading) {
      handleSearch();
    }
  };

  const clearAll = () => {
    setQuery('');
    setLocation('');
    setLastResponse(null);
  };

  const setSuggestion = (suggestion: string) => {
    setQuery(suggestion);
  };

  const setLocationSuggestion = (suggestion: string) => {
    setLocation(suggestion);
  };

  // Travel category filters
  const travelCategories = [
    { icon: '🏛️', label: 'Attractions', query: 'tourist attractions' },
    { icon: '🏨', label: 'Hotels', query: 'hotels' },
    { icon: '🍽️', label: 'Dining', query: 'restaurants' },
    { icon: '🛍️', label: 'Shopping', query: 'shopping centers' },
    { icon: '🚇', label: 'Transport', query: 'public transport hubs' },
    { icon: '🎭', label: 'Entertainment', query: 'entertainment venues' },
    { icon: '☕', label: 'Cafes', query: 'local cafes' },
    { icon: '🏪', label: 'Markets', query: 'local markets' }
  ];

  const setCategoryQuery = (categoryQuery: string) => {
    setQuery(categoryQuery);
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      {/* Main Search Form */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          🔍 Travel Discovery Search
        </Typography>
        
        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', md: 'row' },
          gap: 2,
          mb: 2 
        }}>
          <Box sx={{ flex: { md: '3' } }}>
            <TextField
              fullWidth
              label="What are you looking for?"
              placeholder="e.g., Best sushi restaurants, Coffee shops..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
              InputProps={{
                startAdornment: <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
              }}
            />
          </Box>
          
          <Box sx={{ flex: { md: '2' } }}>
            <TextField
              fullWidth
              label="Location (optional)"
              placeholder="e.g., Paris, San Francisco..."
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
              InputProps={{
                startAdornment: <LocationIcon sx={{ color: 'text.secondary', mr: 1 }} />
              }}
            />
          </Box>
          
          <Box sx={{ flex: { md: '1' } }}>
            <Box sx={{ display: 'flex', gap: 1, height: '100%' }}>
              <Button
                variant="contained"
                onClick={handleSearch}
                disabled={isLoading || !query.trim()}
                sx={{ flex: 1, minHeight: 56 }}
                startIcon={isLoading ? <CircularProgress size={16} /> : <SearchIcon />}
              >
                {isLoading ? 'AI Agent Working...' : 'Search'}
              </Button>
              
              <Tooltip title="Clear all">
                <IconButton
                  onClick={clearAll}
                  disabled={isLoading}
                  sx={{ minHeight: 56 }}
                >
                  <ClearIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>

        {/* Loading Progress */}
        {isLoading && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              AI agent is searching places using local LLM and SerpAPI... This may take 30-60 seconds.
            </Typography>
          </Box>
        )}
      </Box>

      <Divider sx={{ my: 3 }} />

      {/* Travel Categories */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          🌍 Travel Categories
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
          {travelCategories.map((category, index) => (
            <Chip
              key={index}
              label={`${category.icon} ${category.label}`}
              onClick={() => setCategoryQuery(category.query)}
              variant="outlined"
              sx={{ 
                cursor: 'pointer',
                '&:hover': { 
                  backgroundColor: 'primary.light',
                  color: 'primary.contrastText'
                }
              }}
            />
          ))}
        </Box>
      </Box>

      {/* Quick Suggestions */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          💡 Smart Travel Ideas
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          {suggestions.map((suggestion, index) => (
            <Chip
              key={index}
              label={suggestion}
              onClick={() => setSuggestion(suggestion)}
              variant="outlined"
              size="small"
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
        
        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          Popular Locations
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {locationSuggestions.map((suggestion, index) => (
            <Chip
              key={index}
              label={suggestion}
              onClick={() => setLocationSuggestion(suggestion)}
              variant="outlined"
              size="small"
              color="secondary"
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
      </Box>

      {/* Search History */}
      {searchHistory.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recent Searches
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {searchHistory.map((search, index) => (
              <Chip
                key={index}
                label={search}
                onClick={() => {
                  const parts = search.split(' in ');
                  setQuery(parts[0]);
                  if (parts[1]) setLocation(parts[1]);
                }}
                variant="filled"
                size="small"
                color="primary"
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Last Response Summary */}
      {lastResponse && lastResponse.success && (
        <Card sx={{ mt: 3, bgcolor: 'success.main', color: 'success.contrastText' }}>
          <CardContent sx={{ py: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="body2">
                ✅ Found {lastResponse.places.length} result{lastResponse.places.length !== 1 ? 's' : ''} for "{lastResponse.query}"
              </Typography>
              {lastResponse.processing_time && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TimeIcon fontSize="small" />
                  <Typography variant="caption">
                    {lastResponse.processing_time.toFixed(2)}s
                  </Typography>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Paper>
  );
};

export default SearchInterface;