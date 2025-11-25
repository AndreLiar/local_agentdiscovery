import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563eb',
    },
    secondary: {
      main: '#7c3aed',
    },
    background: {
      default: '#f8fafc',
    },
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  },
  shape: {
    borderRadius: 12,
  },
  // Ensure consistent styling across SSR and client
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarGutter: 'stable',
        },
      },
    },
  },
});

export default theme;