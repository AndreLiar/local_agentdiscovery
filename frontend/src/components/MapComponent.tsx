'use client';

import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import { Box } from '@mui/material';

// Import Mapbox CSS
import 'mapbox-gl/dist/mapbox-gl.css';

interface PlaceMarker {
  id: string;
  name: string;
  coordinates: [number, number]; // [lng, lat]
  address?: string;
  rating?: number;
  description?: string;
  price?: string;
  reviews?: number;
  hours?: string;
  type?: string;
  phone?: string;
  website?: string;
}

interface MapComponentProps {
  markers?: PlaceMarker[];
  center?: [number, number];
  zoom?: number;
  onMapLoad?: (map: mapboxgl.Map) => void;
}

const MapComponent: React.FC<MapComponentProps> = ({
  markers = [],
  center = [2.3522, 48.8566], // Default to Paris
  zoom = 12,
  onMapLoad
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return;

    // Set Mapbox access token
    mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN || '';

    if (!mapboxgl.accessToken) {
      console.error('Mapbox access token is required');
      return;
    }

    // Create map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center: center,
      zoom: zoom,
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    // Handle map load
    map.current.on('load', () => {
      setMapLoaded(true);
      if (onMapLoad && map.current) {
        onMapLoad(map.current);
      }
    });

    // Cleanup
    return () => {
      if (map.current) {
        map.current.remove();
      }
    };
  }, [center, zoom, onMapLoad]);

  // Update markers when markers prop changes
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Remove existing markers
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];

    // Add new markers
    markers.forEach((markerData, index) => {
      if (!markerData.coordinates || markerData.coordinates.length !== 2) return;

      // Create marker element
      const markerElement = document.createElement('div');
      markerElement.className = 'marker';
      markerElement.style.cssText = `
        background-color: #3b82f6;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: bold;
      `;
      markerElement.textContent = (index + 1).toString();

      // Create popup content with rich information
      const popupContent = `
        <div style="padding: 12px; min-width: 300px; max-width: 400px; font-family: system-ui, -apple-system, sans-serif;">
          <h3 style="margin: 0 0 8px 0; font-size: 18px; font-weight: bold; color: #1a1a1a;">
            🌟 ${markerData.name}
          </h3>
          
          ${markerData.rating ? `
            <div style="margin-bottom: 6px; color: #ff6600; font-weight: bold; font-size: 16px;">
              ⭐ ${markerData.rating}/5.0
            </div>
          ` : ''}
          
          ${markerData.description ? `
            <div style="margin-bottom: 8px; color: #555; font-size: 14px; line-height: 1.4; font-style: italic;">
              📝 ${markerData.description}
            </div>
          ` : ''}
          
          <div style="display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
            ${markerData.type ? `
              <span style="background: #e3f2fd; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #1976d2;">
                🍽️ ${markerData.type}
              </span>
            ` : ''}
            ${markerData.price ? `
              <span style="background: #f3e5f5; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #7b1fa2;">
                💰 ${markerData.price}
              </span>
            ` : ''}
            ${markerData.reviews ? `
              <span style="background: #e8f5e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #388e3c;">
                📊 ${markerData.reviews.toLocaleString()} reviews
              </span>
            ` : ''}
          </div>
          
          ${markerData.address ? `
            <div style="margin-bottom: 6px; color: #555; font-size: 14px;">
              📍 <strong>Address:</strong> ${markerData.address}
            </div>
          ` : ''}
          
          ${markerData.hours ? `
            <div style="margin-bottom: 6px; color: #555; font-size: 14px;">
              🕐 <strong>Hours:</strong> ${markerData.hours}
            </div>
          ` : ''}
          
          ${markerData.phone ? `
            <div style="margin-bottom: 6px; color: #555; font-size: 14px;">
              📞 <strong>Phone:</strong> ${markerData.phone}
            </div>
          ` : ''}
          
          ${markerData.website ? `
            <div style="margin-bottom: 6px; font-size: 14px;">
              🌐 <strong>Website:</strong> <a href="${markerData.website}" target="_blank" rel="noopener noreferrer" style="color: #1976d2; text-decoration: underline;">Visit Website</a>
            </div>
          ` : ''}
        </div>
      `;

      // Create popup
      const popup = new mapboxgl.Popup({
        offset: 25,
        closeButton: true,
        closeOnClick: true
      }).setHTML(popupContent);

      // Create and add marker
      const marker = new mapboxgl.Marker(markerElement)
        .setLngLat(markerData.coordinates)
        .setPopup(popup)
        .addTo(map.current!);

      markersRef.current.push(marker);
    });

    // Fit map to markers if there are any
    if (markers.length > 0) {
      const bounds = new mapboxgl.LngLatBounds();
      markers.forEach(marker => {
        if (marker.coordinates) {
          bounds.extend(marker.coordinates);
        }
      });
      
      // Only fit bounds if we have valid markers
      if (!bounds.isEmpty()) {
        map.current.fitBounds(bounds, {
          padding: 50,
          maxZoom: 15
        });
      }
    }
  }, [markers, mapLoaded]);

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        borderRadius: 2,
        overflow: 'hidden',
        position: 'relative'
      }}
    >
      <div
        ref={mapContainer}
        style={{
          width: '100%',
          height: '100%'
        }}
      />
    </Box>
  );
};

export default MapComponent;