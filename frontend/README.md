# 🗺️ Local Discovery Agent - Frontend

A beautiful Next.js frontend with Material-UI and Mapbox integration for the Local Discovery Agent.

## ✨ Features

- **🔍 Smart Search Interface** - AI-powered search with suggestions
- **🗺️ Interactive Mapbox Map** - Real-time place plotting with markers
- **📱 Responsive Design** - Works on desktop, tablet, and mobile
- **🎨 Material-UI Components** - Modern, accessible interface
- **⚡ Real-time Updates** - Live backend health monitoring
- **📊 Rich Place Details** - Ratings, addresses, and actions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Setup

Create `.env.local` and add your tokens:

```env
NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 3. Start Development Server

```bash
npm run dev
```

The frontend will be available at **http://localhost:3000**

## 🔧 Configuration

### API Configuration

The frontend connects to the FastAPI backend at:
- **Default**: `http://localhost:8000`
- **Configure**: Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local`

### Mapbox Setup

1. Sign up at [mapbox.com](https://mapbox.com)
2. Create an access token
3. Add to `NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN` in `.env.local`

## 🏗️ Architecture

```
src/
├── app/
│   └── page.tsx              # Main application page
├── components/
│   ├── MapComponent.tsx      # Mapbox map with markers
│   ├── SearchInterface.tsx   # Search form and suggestions
│   └── PlaceResults.tsx      # Results list with actions
└── services/
    └── api.ts               # Backend API integration
```

## 🎯 Main Components

### MapComponent
- Mapbox GL integration with interactive markers
- Auto-fitting bounds for multiple locations
- Rich popups with place details

### SearchInterface
- Real-time search with backend integration
- Quick suggestions and search history
- Location filtering support

### PlaceResults
- Rich place cards with ratings and details
- Map integration and export functionality
- Direct actions (directions, share)

## 📱 Usage

1. **Start Backend**: Make sure the FastAPI backend is running at `localhost:8000`
2. **Configure Mapbox**: Add your Mapbox token to `.env.local`
3. **Start Frontend**: Run `npm run dev`
4. **Search Places**: Enter queries like "sushi restaurants in Paris"
5. **View Results**: See places plotted on the map with detailed information

## 🛠️ Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Run type checking
npx tsc --noEmit
```

## 🌍 Environment Variables

Required environment variables in `.env.local`:

```env
# Mapbox token for map rendering
NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN=pk.your_mapbox_token

# Backend API URL
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## 🚀 Deployment

Deploy to Vercel:

```bash
npm run build
vercel deploy
```

Make sure to set environment variables in your Vercel dashboard.

---

**The frontend provides a beautiful interface for AI-powered local discovery with real-time mapping! 🎉**
