'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Paper,
  Typography,
  Alert,
  Snackbar,
  Box,
  Fab,
  Tooltip,
  IconButton,
  AppBar,
  Toolbar,
  Chip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  MonitorHeart as HealthIcon,
  Map as MapIcon
} from '@mui/icons-material';

import SearchInterface from '@/components/SearchInterface';
import MapComponent from '@/components/MapComponent';
import PlaceResults from '@/components/PlaceResults';
import MemorySelector from '@/components/MemorySelector';
import ClientOnly from '@/components/ClientOnly';
import { PlaceResult, checkHealth } from '@/services/api';
import { theme } from '@/theme/theme';

export default function Home() {
  const [results, setResults] = useState<PlaceResult[]>([]);
  const [currentQuery, setCurrentQuery] = useState<string>('');
  const [selectedPlaceIndex, setSelectedPlaceIndex] = useState<number | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [backendHealth, setBackendHealth] = useState<'healthy' | 'unhealthy' | 'checking'>('checking');
  const [memoryRefreshTrigger, setMemoryRefreshTrigger] = useState<number>(0);

  // Convert PlaceResult to map markers with full rich data
  const mapMarkers = results.map((place, index) => ({
    id: `place-${index}`,
    name: place.name,
    coordinates: place.coordinates || [0, 0] as [number, number],
    address: place.address,
    rating: place.rating,
    description: place.description,
    price: place.price,
    reviews: place.reviews,
    hours: place.hours,
    type: place.type,
    phone: place.phone,
    website: place.website
  })).filter(marker => marker.coordinates[0] !== 0 || marker.coordinates[1] !== 0);

  // Calculate map center
  const mapCenter: [number, number] = React.useMemo(() => {
    if (mapMarkers.length === 0) return [2.3522, 48.8566]; // Default to Paris
    
    const avgLng = mapMarkers.reduce((sum, marker) => sum + marker.coordinates[0], 0) / mapMarkers.length;
    const avgLat = mapMarkers.reduce((sum, marker) => sum + marker.coordinates[1], 0) / mapMarkers.length;
    
    return [avgLng, avgLat];
  }, [mapMarkers]);

  // Handle search results
  const handleResults = useCallback((newResults: PlaceResult[], query: string) => {
    setResults(newResults);
    setCurrentQuery(query);
    setSelectedPlaceIndex(undefined);
    setError(null);
    // Trigger memory refresh to update message count
    setMemoryRefreshTrigger(prev => prev + 1);
  }, []);

  // Handle errors
  const handleError = useCallback((errorMessage: string) => {
    setError(errorMessage);
  }, []);

  // Handle place selection
  const handlePlaceSelect = useCallback((_place: PlaceResult, index: number) => {
    setSelectedPlaceIndex(index);
  }, []);

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
    setBackendHealth('checking');
    try {
      const health = await checkHealth();
      setBackendHealth(health.status === 'healthy' ? 'healthy' : 'unhealthy');
    } catch {
      setBackendHealth('unhealthy');
    }
  }, []);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, [checkBackendHealth]);

  const closeError = () => {
    setError(null);
  };

  return (
    <ClientOnly fallback={<div>Loading...</div>}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        
        {/* App Bar */}
        <AppBar position="static" elevation={0} sx={{ bgcolor: 'background.paper', color: 'text.primary' }}>
          <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
            <MapIcon color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              AI Travel Discovery
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Smart Travel Assistant
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              icon={<HealthIcon />}
              label={backendHealth === 'healthy' ? 'Backend Online' : backendHealth === 'unhealthy' ? 'Backend Offline' : 'Checking...'}
              color={backendHealth === 'healthy' ? 'success' : backendHealth === 'unhealthy' ? 'error' : 'default'}
              size="small"
            />
            <Tooltip title="Refresh backend status">
              <IconButton onClick={checkBackendHealth} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
        <Box sx={{ 
          display: 'flex', 
          gap: 3, 
          height: 'calc(100vh - 120px)',
          flexDirection: { xs: 'column', lg: 'row' }
        }}>
          {/* Left Panel - Search & Results */}
          <Box sx={{ 
            flex: { lg: 5 }, 
            display: 'flex', 
            flexDirection: 'column', 
            gap: 2,
            minHeight: { xs: '50vh', lg: '100%' }
          }}>
            {/* Search Interface */}
            <SearchInterface
              onResults={handleResults}
              onError={handleError}
            />
            
            {/* Memory Selector */}
            <MemorySelector
              onMemoryChange={(memoryType) => {
                console.log('Memory type changed to:', memoryType);
                // Trigger refresh after memory type change
                setMemoryRefreshTrigger(prev => prev + 1);
              }}
              onError={handleError}
              refreshTrigger={memoryRefreshTrigger}
            />
            
            {/* Results Panel */}
            {results.length > 0 && (
              <PlaceResults
                results={results}
                query={currentQuery}
                onPlaceSelect={handlePlaceSelect}
                selectedPlaceIndex={selectedPlaceIndex}
              />
            )}
            
            {/* Travel Welcome Message */}
            {results.length === 0 && (
              <Paper elevation={2} sx={{ p: 4, textAlign: 'center', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <Typography variant="h4" gutterBottom color="primary">
                  ✈️ Welcome to Travel Discovery
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                  Discover amazing places, tourist attractions, hotels, and local experiences around the world using our AI travel assistant.
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  🏛️ Find attractions near landmarks • 🏨 Locate hotels in perfect spots • 🍽️ Discover local dining gems
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Enter your travel query above - our AI understands natural language and finds exactly what you're looking for!
                </Typography>
              </Paper>
            )}
          </Box>

          {/* Right Panel - Map */}
          <Box sx={{ flex: { lg: 7 }, minHeight: { xs: '50vh', lg: '100%' } }}>
            <Paper elevation={3} sx={{ height: '100%', overflow: 'hidden' }}>
              <MapComponent
                markers={mapMarkers}
                center={mapCenter}
                zoom={mapMarkers.length > 0 ? 12 : 10}
              />
            </Paper>
          </Box>
        </Box>
      </Container>

      {/* Error Snackbar */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={closeError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={closeError} severity="error" variant="filled">
          {error}
        </Alert>
      </Snackbar>

      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="refresh"
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1000
        }}
        onClick={checkBackendHealth}
      >
        <RefreshIcon />
      </Fab>
      </ThemeProvider>
    </ClientOnly>
  );
}
