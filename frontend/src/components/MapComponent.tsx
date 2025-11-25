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

      // Create popup content
      const popupContent = `
        <div style="padding: 8px; min-width: 200px;">
          <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: bold; color: #1a1a1a;">
            ${markerData.name}
          </h3>
          ${markerData.rating ? `
            <div style="margin-bottom: 4px; color: #666;">
              ⭐ ${markerData.rating}/5.0
            </div>
          ` : ''}
          ${markerData.address ? `
            <div style="color: #666; font-size: 14px;">
              📍 ${markerData.address}
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