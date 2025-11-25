'use client';

import React from 'react';
import {
  Paper,
  Typography,
  Box
} from '@mui/material';
import { PlaceResult } from '@/services/api';

interface PlaceResultsProps {
  results: PlaceResult[];
  query: string;
  onPlaceSelect?: (place: PlaceResult, index: number) => void;
  selectedPlaceIndex?: number;
}

const PlaceResults: React.FC<PlaceResultsProps> = ({ 
  results, 
  query, 
  onPlaceSelect,
  selectedPlaceIndex 
}) => {
  console.log('🔥 COMPLETELY NEW COMPONENT LOADED 🔥');
  console.log('PlaceResults - Rich data received:', JSON.stringify(results, null, 2));

  if (results.length === 0) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No results found. Try a different search query.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ maxHeight: '70vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, bgcolor: 'error.main', color: 'error.contrastText' }}>
        <Typography variant="h6" gutterBottom>
          🔥 COMPLETELY NEW RICH DATA COMPONENT 🔥
        </Typography>
        <Typography variant="body2">
          Found {results.length} place{results.length !== 1 ? 's' : ''} for "{query}"
        </Typography>
      </Box>

      {/* Results List */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {results.map((place, index) => (
          <div key={index} style={{ 
            marginBottom: '25px', 
            padding: '25px', 
            border: '4px solid purple', 
            borderRadius: '15px', 
            backgroundColor: '#faf0ff',
            boxShadow: '0 6px 12px rgba(128,0,128,0.2)'
          }}>
            <h2 style={{ color: 'purple', fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
              🌟 {index + 1}. {place.name}
            </h2>
            
            <div style={{ fontSize: '18px', marginBottom: '15px', color: '#ff6600' }}>
              ⭐ Rating: {place.rating}/5.0
            </div>
            
            <div style={{ fontSize: '16px', marginBottom: '12px', color: '#333', fontStyle: 'italic' }}>
              📝 <strong>Description:</strong> {place.description || 'No description available'}
            </div>
            
            <div style={{ marginBottom: '15px', backgroundColor: '#f0f0f0', padding: '15px', borderRadius: '8px' }}>
              <div style={{ marginBottom: '8px', fontSize: '16px' }}>
                <span style={{ background: '#e3f2fd', padding: '8px 12px', margin: '4px', borderRadius: '15px', fontSize: '14px', fontWeight: 'bold' }}>
                  🍽️ Type: {place.type || 'Unknown'}
                </span>
              </div>
              <div style={{ marginBottom: '8px', fontSize: '16px' }}>
                <span style={{ background: '#f3e5f5', padding: '8px 12px', margin: '4px', borderRadius: '15px', fontSize: '14px', fontWeight: 'bold' }}>
                  💰 Price: {place.price || 'Not specified'}
                </span>
              </div>
              <div style={{ fontSize: '16px' }}>
                <span style={{ background: '#e8f5e8', padding: '8px 12px', margin: '4px', borderRadius: '15px', fontSize: '14px', fontWeight: 'bold' }}>
                  📊 Reviews: {place.reviews ? place.reviews.toLocaleString() : 'No reviews'}
                </span>
              </div>
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              📍 <strong>Address:</strong> {place.address || 'No address available'}
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              🕐 <strong>Hours:</strong> {place.hours || 'Hours not available'}
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              📏 <strong>Distance:</strong> {place.distance_km ? `${place.distance_km.toFixed(1)} km` : 'Distance not calculated'}
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              📞 <strong>Phone:</strong> {place.phone || 'Phone not available'}
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              🌐 <strong>Website:</strong> {place.website ? 
                <a href={place.website} target="_blank" rel="noopener noreferrer" style={{ color: '#1976d2', textDecoration: 'underline' }}>
                  {place.website}
                </a> : 
                'Website not available'
              }
            </div>
            
            <div style={{ fontSize: '15px', marginBottom: '8px', color: '#555' }}>
              🗺️ <strong>Coordinates:</strong> {
                place.coordinates && place.coordinates[0] !== 0 && place.coordinates[1] !== 0 ? 
                `Lat: ${place.coordinates[1].toFixed(4)}, Lng: ${place.coordinates[0].toFixed(4)}` : 
                'Coordinates not available'
              }
            </div>
          </div>
        ))}
      </Box>
    </Paper>
  );
};

export default PlaceResults;