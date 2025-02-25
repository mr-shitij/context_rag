import React, { useEffect, useRef } from 'react';
import { Send, Loader, User, Bot, Menu } from 'lucide-react';
import { Message } from '../types';
import { format } from 'date-fns';
import clsx from 'clsx';

interface ChatContainerProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  onToggleSidebar?: () => void;
}

export function ChatContainer({ messages, isLoading, onSendMessage, onToggleSidebar }: ChatContainerProps) {
  const [input, setInput] = React.useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-[100dvh]">
      <div className="bg-white border-b p-4 md:hidden">
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-gray-100 rounded-lg"
        >
          <Menu className="w-6 h-6" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        {messages.map((message) => (
          <div
  key={message.id}
  className={clsx(
    'mb-6 mx-auto w-full max-w-[90%]', // Ensure messages do not exceed the screen width
    'px-2 sm:px-4',
    message.role === 'user' ? 'ml-auto' : 'mr-auto'
  )}
>

            <div className={clsx(
              'flex items-start gap-3',
              message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
            )}>
              <div className={clsx(
                'w-8 h-8 rounded-full flex items-center justify-center',
                message.role === 'user' ? 'bg-indigo-600' : 'bg-gray-600'
              )}>
                {message.role === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>
              <div className={clsx(
                'rounded-2xl p-4 shadow-sm',
                message.role === 'user' 
                  ? 'bg-indigo-600 text-white rounded-tr-none' 
                  : 'bg-white text-gray-900 rounded-tl-none border border-gray-200'
              )}>
                <p className="whitespace-pre-wrap">{message.content}</p>
                <div className={clsx(
                  'text-xs mt-2',
                  message.role === 'user' ? 'text-indigo-200' : 'text-gray-500'
                )}>
                  {format(message.timestamp, 'h:mm a')}
                </div>
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex items-center justify-center py-4">
            <Loader className="animate-spin text-gray-500" size={24} />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t p-4 bg-white sticky bottom-0">
        <div className="max-w-3xl mx-auto flex gap-4 px-2 sm:px-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={20} />
          </button>
        </div>
      </form>
    </div>
  );
}