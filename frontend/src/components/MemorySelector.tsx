'use client';

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Button,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Memory as MemoryIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Info as InfoIcon,
  Psychology as BrainIcon,
  Schedule as ClockIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

import { 
  getMemoryInfo, 
  getMemoryTypes, 
  switchMemoryType, 
  MemoryInfo, 
  MemoryType 
} from '@/services/api';

interface MemorySelectorProps {
  onMemoryChange?: (memoryType: string) => void;
  onError?: (error: string) => void;
}

const MemorySelector: React.FC<MemorySelectorProps> = ({ onMemoryChange, onError }) => {
  const [currentMemory, setCurrentMemory] = useState<MemoryInfo | null>(null);
  const [memoryTypes, setMemoryTypes] = useState<MemoryType[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [isChanging, setIsChanging] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  // Load current memory info and available types
  useEffect(() => {
    loadMemoryData();
  }, []);

  const loadMemoryData = async () => {
    setIsLoading(true);
    try {
      const [memoryInfo, typesData] = await Promise.all([
        getMemoryInfo(),
        getMemoryTypes()
      ]);
      
      setCurrentMemory(memoryInfo);
      setMemoryTypes(typesData.memory_types);
      setSelectedType(memoryInfo.type);
    } catch (error) {
      onError?.(error instanceof Error ? error.message : 'Failed to load memory data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleMemoryTypeChange = async () => {
    if (!selectedType || selectedType === currentMemory?.type) return;

    setIsChanging(true);
    try {
      await switchMemoryType(selectedType);
      await loadMemoryData(); // Reload to get updated info
      onMemoryChange?.(selectedType);
      setDialogOpen(false);
    } catch (error) {
      onError?.(error instanceof Error ? error.message : 'Failed to switch memory type');
    } finally {
      setIsChanging(false);
    }
  };

  const getMemoryIcon = (type: string) => {
    switch (type) {
      case 'buffer':
        return <StorageIcon />;
      case 'window':
        return <ClockIcon />;
      case 'summary':
        return <BrainIcon />;
      default:
        return <MemoryIcon />;
    }
  };

  const getMemoryColor = (type: string) => {
    switch (type) {
      case 'buffer':
        return 'primary';
      case 'window':
        return 'secondary';
      case 'summary':
        return 'success';
      default:
        return 'default';
    }
  };

  if (isLoading) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <CircularProgress size={24} />
        <Typography variant="body2" sx={{ mt: 1 }}>
          Loading memory information...
        </Typography>
      </Paper>
    );
  }

  return (
    <>
      <Paper elevation={2} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <MemoryIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            Conversational Memory
          </Typography>
          <Tooltip title="Configure how the AI remembers conversation context">
            <IconButton size="small" onClick={() => setDialogOpen(true)}>
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {currentMemory && (
          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent sx={{ py: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  {getMemoryIcon(currentMemory.type)}
                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      {memoryTypes.find(t => t.type === currentMemory.type)?.name || 'Current Memory'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {currentMemory.description}
                    </Typography>
                  </Box>
                </Box>
                <Chip
                  label={`${currentMemory.message_count} messages`}
                  size="small"
                  color={getMemoryColor(currentMemory.type) as any}
                  variant="outlined"
                />
              </Box>
            </CardContent>
          </Card>
        )}

        <Button
          variant="outlined"
          startIcon={<MemoryIcon />}
          onClick={() => setDialogOpen(true)}
          fullWidth
        >
          Configure Memory Type
        </Button>
      </Paper>

      {/* Memory Configuration Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <BrainIcon />
            Configure Conversational Memory
          </Box>
        </DialogTitle>
        
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            Choose how the AI agent remembers conversation context. This affects performance and memory usage.
          </Alert>

          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend" sx={{ mb: 2 }}>
              Select Memory Type
            </FormLabel>
            
            <RadioGroup
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
            >
              {memoryTypes.map((memoryType) => (
                <Box key={memoryType.type} sx={{ mb: 2 }}>
                  <FormControlLabel
                    value={memoryType.type}
                    control={<Radio />}
                    label={
                      <Box sx={{ ml: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getMemoryIcon(memoryType.type)}
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            {memoryType.name}
                          </Typography>
                          {currentMemory?.type === memoryType.type && (
                            <Chip label="Current" size="small" color="primary" />
                          )}
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          {memoryType.description}
                        </Typography>
                        
                        <Box sx={{ mt: 1, display: 'flex', gap: 2 }}>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              Pros:
                            </Typography>
                            <List dense sx={{ py: 0 }}>
                              {memoryType.pros.map((pro, index) => (
                                <ListItem key={index} sx={{ py: 0, px: 0 }}>
                                  <ListItemIcon sx={{ minWidth: 20 }}>
                                    <CheckIcon fontSize="small" color="success" />
                                  </ListItemIcon>
                                  <ListItemText 
                                    primary={pro} 
                                    primaryTypographyProps={{ variant: 'caption' }}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Box>
                          
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              Cons:
                            </Typography>
                            <List dense sx={{ py: 0 }}>
                              {memoryType.cons.map((con, index) => (
                                <ListItem key={index} sx={{ py: 0, px: 0 }}>
                                  <ListItemIcon sx={{ minWidth: 20 }}>
                                    <CancelIcon fontSize="small" color="warning" />
                                  </ListItemIcon>
                                  <ListItemText 
                                    primary={con} 
                                    primaryTypographyProps={{ variant: 'caption' }}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Box>
                        </Box>
                      </Box>
                    }
                    sx={{ 
                      alignItems: 'flex-start',
                      border: selectedType === memoryType.type ? 2 : 1,
                      borderColor: selectedType === memoryType.type ? 'primary.main' : 'divider',
                      borderRadius: 2,
                      p: 2,
                      m: 0,
                      '&:hover': {
                        bgcolor: 'action.hover'
                      }
                    }}
                  />
                  {memoryType.type !== memoryTypes[memoryTypes.length - 1].type && (
                    <Divider sx={{ mt: 2 }} />
                  )}
                </Box>
              ))}
            </RadioGroup>
          </FormControl>
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            variant="contained"
            onClick={handleMemoryTypeChange}
            disabled={isChanging || selectedType === currentMemory?.type}
            startIcon={isChanging ? <CircularProgress size={16} /> : null}
          >
            {isChanging ? 'Applying...' : 'Apply Changes'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default MemorySelector;