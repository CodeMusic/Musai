'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ArrowLeft, CheckCircle } from 'lucide-react';
import { motion } from 'motion/react';

export default function RoverBytePage() {
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Here you would typically send the email to your backend
    console.log('Email submitted:', email);
    setIsSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background/95 to-secondary/10 dark:to-primary/5">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="mb-8">
          <Link 
            href="/" 
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Return to Musai
          </Link>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto text-center space-y-12">
          {/* Logos */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center gap-8 flex-wrap"
          >
            <div className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Musai</div>
            <div className="text-2xl text-muted-foreground">×</div>
            <div className="text-4xl font-bold bg-gradient-to-r from-secondary to-primary bg-clip-text text-transparent">RoverByte</div>
          </motion.div>

          {/* Title and Subtitle */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="space-y-4"
          >
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              Musai <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">×</span> RoverByte
            </h1>
            <h2 className="text-2xl md:text-3xl text-muted-foreground font-medium">
              Merging Minds, Machines & Meaning
            </h2>
          </motion.div>

          {/* Description */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="max-w-2xl mx-auto"
          >
            <p className="text-lg md:text-xl text-muted-foreground leading-relaxed">
              Musai is evolving with RoverByte—a comprehensive automation ecosystem where Musai serves as the central AI task solution, orchestrated through n8n workflows to connect RoverByte and all integrated services seamlessly.
            </p>
          </motion.div>

          {/* Email Signup */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="max-w-md mx-auto"
          >
            {!isSubmitted ? (
              <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="flex-1"
                />
                <Button type="submit" size="lg">
                  Notify Me
                </Button>
              </form>
            ) : (
              <div className="flex items-center justify-center gap-2 text-green-600">
                <CheckCircle className="h-5 w-5" />
                <span className="font-medium">Thanks! We'll keep you updated.</span>
              </div>
            )}
          </motion.div>

          {/* Roadmap */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="max-w-3xl mx-auto"
          >
            <h3 className="text-2xl font-bold mb-8">Coming Soon</h3>
            <div className="grid md:grid-cols-3 gap-6 text-left">
              <div className="bg-card/50 backdrop-blur border rounded-lg p-6 space-y-3 border-secondary/20">
                <div className="text-sm text-primary font-semibold">Phase 1</div>
                <h4 className="font-semibold">Musai Core & n8n Integration</h4>
                <p className="text-sm text-muted-foreground">
                  Deploy Musai as the central AI task solution with n8n workflow orchestration
                </p>
              </div>
              <div className="bg-card/50 backdrop-blur border rounded-lg p-6 space-y-3 border-primary/20">
                <div className="text-sm text-secondary font-semibold">Phase 2</div>
                <h4 className="font-semibold">RoverByte Connection</h4>
                <p className="text-sm text-muted-foreground">
                  Connect RoverByte and ecosystem services through Musai's workflow automation
                </p>
              </div>
              <div className="bg-card/50 backdrop-blur border rounded-lg p-6 space-y-3 border-accent/20">
                <div className="text-sm text-accent font-semibold">Phase 3</div>
                <h4 className="font-semibold">Ecosystem Unification</h4>
                <p className="text-sm text-muted-foreground">
                  Complete unified ecosystem with Musai orchestrating all services via n8n workflows
                </p>
              </div>
            </div>
          </motion.div>

          {/* Links */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="space-y-4"
          >
            <div className="text-sm text-muted-foreground">
              Learn more about RoverByte:
            </div>
            <Link 
              href="https://roverbyte.codemusic.ca" 
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-primary hover:underline font-medium"
            >
              Visit RoverByte
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
} 