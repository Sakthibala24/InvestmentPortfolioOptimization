import React from 'react';
import { useBackendConnection } from '../hooks/useBackendConnection';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';

export const ConnectionStatus: React.FC = () => {
  const { isConnected, isLoading, error, checkConnection } = useBackendConnection();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 rounded-full text-sm">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Checking backend...</span>
      </div>
    );
  }

  if (isConnected) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 rounded-full text-sm">
        <Wifi className="w-4 h-4" />
        <span>Backend Connected</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 rounded-full text-sm">
      <WifiOff className="w-4 h-4" />
      <span>Backend Offline</span>
      <button
        onClick={checkConnection}
        className="ml-2 px-2 py-1 bg-red-200 dark:bg-red-800 hover:bg-red-300 dark:hover:bg-red-700 rounded text-xs transition-colors"
      >
        Retry
      </button>
    </div>
  );
};