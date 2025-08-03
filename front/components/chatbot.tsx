"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import ReactMarkdown from "react-markdown"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Bot,
  User,
  Send,
  Home,
  TrendingUp,
  Bell,
  AlertTriangle,
  Calendar,
  MapPin,
  Clock,
  X,
  ChevronRight,
  Building,
  Key,
  DollarSign,
} from "lucide-react"

const WS_URL = "ws://srv889806.hstgr.cloud/ws/chat/"

interface Message {
  sender: "user" | "bot"
  text: string
  timestamp?: Date
  showAlertsButton?: boolean
}

interface Alert {
  id: string
  type: "maintenance" | "urgent" | "inspection" | "payment"
  title: string
  property: string
  address: string
  action: string
  riskFactors: string
  lastInspection: string
  priority: "high" | "medium" | "low"
  timestamp: Date
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "bot",
      text: "Hi! I am your AI Assistant for Landlords. Ask me about rent esitmation, tenant screening, or check Property Maintaince Alerts!",
      timestamp: null,
    },
  ])
  const [input, setInput] = useState("")
  const [isConnected, setIsConnected] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [isAlertsOpen, setIsAlertsOpen] = useState(false)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [isClient, setIsClient] = useState(false)
  const [selectedIntent, setSelectedIntent] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Fix hydration issue by ensuring client-side rendering for timestamps
  useEffect(() => {
    setIsClient(true)
    // Fix hydration: set initial message timestamp on client
    setMessages((msgs) =>
      msgs.map((msg, idx) =>
        idx === 0 && !msg.timestamp ? { ...msg, timestamp: new Date() } : msg
      )
    )
  }, [])

  useEffect(() => {
    ws.current = new WebSocket(WS_URL)

    ws.current.onopen = () => {
      setIsConnected(true)
      // Fetch alerts on connect
      ws.current?.send(JSON.stringify({ type: "get_alerts" }))
    }
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setIsTyping(false)

      if (data.type === "alerts") {
        // Backend returns alerts as an array of alert objects
        setAlerts(
          (data.alerts || []).map((alert: any) => ({
            ...alert,
            timestamp: alert.timestamp ? new Date(alert.timestamp) : new Date(),
          })),
        )
        return
      }
      if (data.type === "property_response") {
        setMessages((msgs) => [
          ...msgs,
          {
            sender: "bot",
            text:
              data.explanation ||
              `🎯 **Predicted Rent:** £${data.predicted_rent}\n\n💡 **Rent Range:** £${data.lower_rent}–£${data.upper_rent}\n\n🔒 **Confidence:** ${data.confidence_percentage}%\n\n📝 ${data.explanation}`,
            timestamp: new Date(),
          },
        ])
      } else if (data.type === "bot_response") {
        const shouldShowAlertsButton =
          data.message.toLowerCase().includes("alert") ||
          data.message.toLowerCase().includes("notification") ||
          data.message.toLowerCase().includes("maintenance")

        setMessages((msgs) => [
          ...msgs,
          {
            sender: "bot",
            text: data.message,
            timestamp: new Date(),
            showAlertsButton: shouldShowAlertsButton,
          },
        ])
      } else if (data.type === "echo") {
        setMessages((msgs) => [...msgs, { sender: "bot", text: data.message, timestamp: new Date() }])
      } else if (data.type === "error") {
        setMessages((msgs) => [...msgs, { sender: "bot", text: `❌ **Error:** ${data.error}`, timestamp: new Date() }])
      }
    }
    ws.current.onclose = () => {
      setIsConnected(false)
    }
    return () => ws.current?.close()
  }, [])

  // Fetch alerts again when sidebar is opened
  useEffect(() => {
    if (isAlertsOpen && ws.current?.readyState === 1) {
      ws.current.send(JSON.stringify({ type: "get_alerts" }))
    }
  }, [isAlertsOpen])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !ws.current) return

    setMessages((msgs) => [...msgs, { sender: "user", text: input, timestamp: new Date() }])
    setIsTyping(true)

    // Send both input and selectedIntent to backend
    ws.current.send(
      JSON.stringify({
        type: "text",
        message: input,
        intent: selectedIntent, // null if not selected
      }),
    )

    setInput("")
    // Optionally unselect intent after send:
    // setSelectedIntent(null)
  }

  // Intent button handler (toggle)
  const handleIntent = (intent: string) => {
    // Use backend intent keys
    const intentMap: Record<string, string> = {
      rent: "rent_prediction",
      tenant: "tenant_screening",
      maintenance: "maintenance_prediction",
    };
    setSelectedIntent((prev) => (prev === intentMap[intent] ? null : intentMap[intent]));
  }

  const formatTime = (date: Date) => {
    if (!isClient) return "" // Prevent hydration mismatch
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "maintenance":
        return <AlertTriangle className="h-4 w-4" />
      case "urgent":
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      case "inspection":
        return <Calendar className="h-4 w-4" />
      default:
        return <Bell className="h-4 w-4" />
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200"
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "low":
        return "bg-green-100 text-green-800 border-green-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const urgentAlerts = alerts.filter((alert) => alert.priority === "high").length

  // Add a helper to detect if a message is a rent prediction (by type or content)
  const isRentPrediction = (msg: Message) => {
    // Heuristic: check for 'Predicted Rent' or similar in bot message
    return (
      msg.sender === "bot" &&
      (msg.text.includes("Predicted Rent") || msg.text.includes("Rent Range") || msg.text.includes("Confidence"))
    )
  }

  // Track the last rent prediction message and its index
  const lastRentPredictionIdx = messages
    .map((msg, idx) => (isRentPrediction(msg) ? idx : -1))
    .filter((idx) => idx !== -1)
    .pop()

  // Add support for 'compare & save' follow-up
  const handleFollowup = async (action: "compare" | "save" | "both") => {
    if (!ws.current || lastRentPredictionIdx == null) return
    ws.current.send(
      JSON.stringify({
        type: "followup",
        action,
      })
    )
    setIsTyping(true)
  }

  // Helper: Parse similar listings from bot message
  function parseSimilarListings(text: string) {
    // Improved regex: Address: (greedy, up to ', Bedrooms:'), Bedrooms: ..., Bathrooms: ..., Size: ..., Property Type: ..., Rent: £...
    const lines = text.split("\n").filter((l) => l.trim().startsWith("- Address:"));
    const listings = lines.map((line) => {
      const addrMatch = line.match(/Address: (.*?), Bedrooms:/);
      const bedsMatch = line.match(/Bedrooms: ([^,]+),/);
      const bathsMatch = line.match(/Bathrooms: ([^,]+),/);
      const sizeMatch = line.match(/Size: ([^,]+) sq ft/);
      const ptypeMatch = line.match(/Property Type: ([^,]+),?/);
      const rentMatch = line.match(/Rent: £([\d.]+)/);
      return {
        address: addrMatch ? addrMatch[1].trim() : "",
        bedrooms: bedsMatch ? bedsMatch[1].trim() : "",
        bathrooms: bathsMatch ? bathsMatch[1].trim() : "",
        size: sizeMatch ? sizeMatch[1].trim() : "",
        propertyType: ptypeMatch ? ptypeMatch[1].trim() : "",
        rent: rentMatch ? rentMatch[1].trim() : "",
      };
    });
    return listings.length > 0 ? listings : null;
  }

  // Helper: Extract summary (first 1-2 lines before listings)
  function extractSummary(text: string) {
    const lines = text.split("\n");
    const summaryLines = [];
    for (const line of lines) {
      if (line.trim().startsWith("- Address:")) break;
      if (line.trim() !== "") summaryLines.push(line);
      if (summaryLines.length >= 2) break;
    }
    return summaryLines.join(" ");
  }

  return (
    <div className="h-screen w-screen relative overflow-hidden">
      {/* Enhanced Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900"></div>

      {/* Animated Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-tr from-blue-600/20 via-purple-600/20 to-pink-600/20 animate-pulse"></div>

      {/* Grid Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
          `,
            backgroundSize: "50px 50px",
          }}
        ></div>
      </div>

      {/* Floating Geometric Shapes */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Large floating circles */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-blue-400/10 to-purple-400/10 rounded-full mix-blend-multiply filter blur-xl animate-float"></div>
        <div className="absolute top-3/4 right-1/4 w-80 h-80 bg-gradient-to-r from-purple-400/10 to-pink-400/10 rounded-full mix-blend-multiply filter blur-xl animate-float-delayed"></div>
        <div className="absolute top-1/2 left-3/4 w-64 h-64 bg-gradient-to-r from-pink-400/10 to-blue-400/10 rounded-full mix-blend-multiply filter blur-xl animate-float-slow"></div>

        {/* Property-themed floating icons */}
        <div className="absolute top-20 left-20 text-white/5 animate-float">
          <Building className="h-24 w-24" />
        </div>
        <div className="absolute top-40 right-32 text-white/5 animate-float-delayed">
          <Home className="h-20 w-20" />
        </div>
        <div className="absolute bottom-32 left-32 text-white/5 animate-float-slow">
          <Key className="h-16 w-16" />
        </div>
        <div className="absolute bottom-20 right-20 text-white/5 animate-float">
          <DollarSign className="h-18 w-18" />
        </div>

        {/* Smaller decorative elements */}
        <div className="absolute top-1/3 left-1/2 w-4 h-4 bg-blue-400/20 rounded-full animate-ping"></div>
        <div className="absolute top-2/3 left-1/3 w-3 h-3 bg-purple-400/20 rounded-full animate-ping delay-1000"></div>
        <div className="absolute top-1/2 right-1/3 w-2 h-2 bg-pink-400/20 rounded-full animate-ping delay-2000"></div>
      </div>

      {/* Diagonal Lines Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.1) 2px,
            rgba(255,255,255,0.1) 4px
          )`,
          }}
        ></div>
      </div>

      {/* Notification Button removed */}

      <div className="h-full flex items-center justify-center p-4 relative z-10">
        <div className={`transition-all duration-300 w-full max-w-4xl h-full ${isAlertsOpen ? "mr-96" : ""}`}>
          <Card className="h-full flex flex-col shadow-2xl border-0 bg-white/95 backdrop-blur-lg">
            <CardHeader className="border-b bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg relative overflow-hidden flex-shrink-0">
              {/* Header background pattern */}
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600/90 to-purple-600/90"></div>
              <div
                className="absolute inset-0 opacity-20"
                style={{
                  backgroundImage: `repeating-linear-gradient(
                  45deg,
                  transparent,
                  transparent 10px,
                  rgba(255,255,255,0.1) 10px,
                  rgba(255,255,255,0.1) 20px
                )`,
                }}
              ></div>

              <div className="flex items-center justify-between relative z-10">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-white/20 rounded-full backdrop-blur-sm">
                    <Home className="h-6 w-6" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-bold">AI for Landlords</CardTitle>
                    <p className="text-blue-100 text-sm">Your intelligent property assistant</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={isConnected ? "default" : "destructive"} className="bg-white/20 backdrop-blur-sm">
                    <div
                      className={`w-2 h-2 rounded-full mr-2 ${isConnected ? "bg-green-400" : "bg-red-400"} ${isConnected ? "animate-pulse" : ""}`}
                    />
                    {isConnected ? "Connected" : "Disconnected"}
                  </Badge>
                </div>
              </div>
            </CardHeader>

            <CardContent className="flex-1 p-0 overflow-hidden">
              <ScrollArea className="h-full">
                <div className="p-6 space-y-6">
                  {messages.map((msg, idx) => {
                    // Check if this is a similar listings message
                    const similarListings = msg.sender === "bot" ? parseSimilarListings(msg.text) : null;
                    const summary = similarListings ? extractSummary(msg.text) : null;
                    return (
                      <div key={idx}>
                        <div
                          className={`flex items-start space-x-3 ${
                            msg.sender === "user" ? "flex-row-reverse space-x-reverse" : ""
                          }`}
                        >
                          <Avatar className="w-10 h-10 border-2 border-white shadow-lg flex-shrink-0">
                            {msg.sender === "user" ? (
                              <>
                                <AvatarImage src="/placeholder-logo.avif" />
                                <AvatarFallback className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
                                  <User className="h-5 w-5" />
                                </AvatarFallback>
                              </>
                            ) : (
                              <>
                                <AvatarImage src="/chatbot.avif" />
                                <AvatarFallback className="bg-gradient-to-r from-purple-500 to-blue-500 text-white">
                                  <Bot className="h-5 w-5" />
                                </AvatarFallback>
                              </>
                            )}
                          </Avatar>

                          <div className={`flex-1 max-w-[80%] ${msg.sender === "user" ? "text-right" : ""}`}>
                            <div className={`inline-block p-4 rounded-2xl shadow-lg ${msg.sender === "user" ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-br-md" : "bg-white/80 backdrop-blur-sm text-gray-800 rounded-bl-md border border-gray-200/50"}`}>
                              {similarListings && similarListings.length >= 2 ? (
                                <div>
                                  {summary && <div className="mb-2 text-sm text-gray-700 font-medium">{summary}</div>}
                                  <div className="overflow-x-auto w-full">
                                    <table className="min-w-[700px] border border-gray-200 rounded-lg text-sm">
                                      <thead className="bg-blue-50">
                                        <tr>
                                          <th className="px-3 py-2 text-left font-semibold w-2/5 min-w-[160px]">Address</th>
                                          <th className="px-3 py-2 text-left font-semibold">Bedrooms</th>
                                          <th className="px-3 py-2 text-left font-semibold">Bathrooms</th>
                                          <th className="px-3 py-2 text-left font-semibold">Size (sq ft)</th>
                                          {/* <th className="px-3 py-2 text-left font-semibold">Property Type</th> */}
                                          <th className="px-3 py-2 text-left font-semibold">Rent (£)</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {similarListings.map((listing, i) => (
                                          <tr key={i} className="border-t border-gray-100 hover:bg-blue-50/50">
                                            <td className="px-3 py-2">{listing.address}</td>
                                            <td className="px-3 py-2">{listing.bedrooms}</td>
                                            <td className="px-3 py-2">{listing.bathrooms}</td>
                                            <td className="px-3 py-2">{listing.size}</td>
                                            {/* <td className="px-3 py-2">{listing.propertyType}</td> */}
                                            <td className="px-3 py-2">{listing.rent}</td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                </div>
                              ) : (
                                msg.sender === "bot" ? (
                                  <div className="prose prose-sm max-w-none">
                                    <ReactMarkdown
                                      components={{
                                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                                        strong: ({ children }) => (
                                          <strong className="font-semibold text-gray-900">{children}</strong>
                                        ),
                                      }}
                                    >
                                      {msg.text}
                                    </ReactMarkdown>
                                  </div>
                                ) : (
                                  <div className="whitespace-pre-wrap">{msg.text}</div>
                                )
                              )}
                            </div>
                            {msg.timestamp && isClient && (
                              <div className={`text-xs text-gray-400 mt-1 ${msg.sender === "user" ? "text-right" : ""}`}>
                                {formatTime(msg.timestamp)}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Alerts Button removed */}

                        {/* Follow-up buttons for rent prediction */}
                        {isRentPrediction(msg) && idx === lastRentPredictionIdx && (
                          <div className="mt-3 ml-14 flex gap-3 flex-wrap">
                            <Button
                              variant="outline"
                              className="bg-green-50/80 border-green-200 text-green-700 hover:bg-green-100/80 shadow-md hover:shadow-lg"
                              onClick={() => handleFollowup("compare")}
                            >
                              Compare to similar listings
                            </Button>
                            <Button
                              variant="outline"
                              className="bg-blue-50/80 border-blue-200 text-blue-700 hover:bg-blue-100/80 shadow-md hover:shadow-lg"
                              onClick={() => handleFollowup("save")}
                            >
                              Save this property
                            </Button>
                            <Button
                              variant="outline"
                              className="bg-purple-50/80 border-purple-200 text-purple-700 hover:bg-purple-100/80 shadow-md hover:shadow-lg"
                              onClick={() => handleFollowup("both")}
                            >
                              Compare & Save
                            </Button>
                          </div>
                        )}
                      </div>
                    );
                  })}

                  {isTyping && (
                    <div className="flex items-start space-x-3">
                      <Avatar className="w-10 h-10 border-2 border-white shadow-lg flex-shrink-0">
                        <AvatarFallback className="bg-gradient-to-r from-purple-500 to-blue-500 text-white">
                          <Bot className="h-5 w-5" />
                        </AvatarFallback>
                      </Avatar>
                      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl rounded-bl-md p-4 shadow-lg">
                        <div className="flex items-center space-x-2">
                          <div className="flex space-x-1">
                            <div
                              className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "0ms" }}
                            />
                            <div
                              className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "150ms" }}
                            />
                            <div
                              className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "300ms" }}
                            />
                          </div>
                          <span className="text-sm text-gray-500 ml-2">AI is thinking...</span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
            </CardContent>

            <div className="border-t bg-white/50 backdrop-blur-sm p-6 flex-shrink-0">
              <form onSubmit={handleSend} className="flex space-x-3">
                <div className="flex-1 relative">
                  <Input
                    type="text"
                    placeholder="Type your message or rent prediction..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="pr-12 h-12 border-2 border-gray-200/50 focus:border-blue-400 rounded-xl bg-white/80 backdrop-blur-sm shadow-md"
                    disabled={!isConnected || isTyping}
                  />
                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                    <TrendingUp className="h-4 w-4 text-gray-400" />
                  </div>
                </div>
                <Button
                  type="submit"
                  disabled={!input.trim() || !isConnected || isTyping}
                  className="h-12 px-6 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 rounded-xl shadow-lg transition-all duration-200 hover:shadow-xl disabled:opacity-50 hover:scale-105"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </form>

              {/* Intent Buttons Below Input */}
              <div className="flex justify-center gap-4 mt-4 mb-2">
                <Button
                  variant={selectedIntent === "rent_prediction" ? "default" : "secondary"}
                  className={`rounded-full px-6 py-2 shadow-md hover:scale-105 transition-all ${selectedIntent === "rent_prediction" ? "ring-2 ring-blue-500" : ""}`}
                  onClick={() => handleIntent("rent")}
                >
                  🏠 Rent Estimation
                </Button>
                <Button
                  variant={selectedIntent === "tenant_screening" ? "default" : "secondary"}
                  className={`rounded-full px-6 py-2 shadow-md hover:scale-105 transition-all ${selectedIntent === "tenant_screening" ? "ring-2 ring-purple-500" : ""}`}
                  onClick={() => handleIntent("tenant")}
                >
                  👤 Tenant Screening
                </Button>
                <Button
                  variant={selectedIntent === "maintenance_prediction" ? "default" : "secondary"}
                  className={`rounded-full px-6 py-2 shadow-md hover:scale-105 transition-all ${selectedIntent === "maintenance_prediction" ? "ring-2 ring-green-500" : ""}`}
                  onClick={() => handleIntent("maintenance")}
                >
                  🛠️ Maintenance Prediction
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Alerts Sidebar removed */}

      {/* Enhanced Overlay removed */}
    </div>
  )
}
