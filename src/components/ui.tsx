import React from 'react';

// Card Components
export const Card = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 ${className}`}>
    {children}
  </div>
);

export const CardContent = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 ${className}`}>
    {children}
  </div>
);

// Button Component
export const Button = ({ 
  children, 
  onClick, 
  className = "", 
  ...props 
}: { 
  children: React.ReactNode; 
  onClick?: () => void; 
  className?: string; 
  [key: string]: any; 
}) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium transition-colors duration-200 ${className}`}
    {...props}
  >
    {children}
  </button>
);

// Input Component
export const Input = ({ 
  label, 
  className = "", 
  ...props 
}: { 
  label?: string; 
  className?: string; 
  [key: string]: any; 
}) => (
  <div className="flex flex-col space-y-1">
    {label && <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>}
    <input
      className={`px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${className}`}
      {...props}
    />
  </div>
);

// Select Components
export const Select = ({ 
  children, 
  label, 
  value, 
  onValueChange, 
  className = "" 
}: { 
  children: React.ReactNode; 
  label?: string; 
  value: string; 
  onValueChange: (value: string) => void; 
  className?: string; 
}) => (
  <div className="flex flex-col space-y-1">
    {label && <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>}
    <select
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
      className={`px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${className}`}
    >
      {children}
    </select>
  </div>
);

export const SelectItem = ({ value, children }: { value: string; children: React.ReactNode }) => (
  <option value={value}>{children}</option>
);

// Tabs Components
export const Tabs = ({ 
  children, 
  value, 
  onValueChange, 
  className = "" 
}: { 
  children: React.ReactNode; 
  value: string; 
  onValueChange: (value: string) => void; 
  className?: string; 
}) => (
  <div className={`${className}`}>
    <div className="flex space-x-1 border-b border-gray-200 dark:border-gray-700">
      {React.Children.map(children, (child) =>
        React.cloneElement(child as React.ReactElement, { activeTab: value, onTabChange: onValueChange })
      )}
    </div>
  </div>
);

export const Tab = ({ 
  value, 
  title, 
  activeTab, 
  onTabChange 
}: { 
  value: string; 
  title: string; 
  activeTab?: string; 
  onTabChange?: (value: string) => void; 
}) => (
  <button
    onClick={() => onTabChange && onTabChange(value)}
    className={`px-4 py-2 font-medium text-sm rounded-t-md transition-colors duration-200 ${
      activeTab === value
        ? 'bg-blue-600 text-white border-b-2 border-blue-600'
        : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
    }`}
  >
    {title}
  </button>
);

// Table Components
export const Table = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className="overflow-x-auto">
    <table className={`min-w-full divide-y divide-gray-200 dark:divide-gray-700 ${className}`}>
      {children}
    </table>
  </div>
);

export const TableHeader = ({ children }: { children: React.ReactNode }) => (
  <thead className="bg-gray-50 dark:bg-gray-800">
    <tr>{children}</tr>
  </thead>
);

export const TableColumn = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <th className={`px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider ${className}`}>
    {children}
  </th>
);

export const TableBody = ({ children }: { children: React.ReactNode }) => (
  <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
    {children}
  </tbody>
);

export const TableRow = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <tr className={`hover:bg-gray-50 dark:hover:bg-gray-800 ${className}`}>
    {children}
  </tr>
);

export const TableCell = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <td className={`px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 ${className}`}>
    {children}
  </td>
);