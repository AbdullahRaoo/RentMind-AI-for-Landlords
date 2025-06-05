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

const WS_URL = "ws://localhost:8000/ws/chat/"

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

    // Check if user is asking about alerts
    const isAlertQuery =
      input.toLowerCase().includes("alert") ||
      input.toLowerCase().includes("notification") ||
      input.toLowerCase().includes("maintenance") ||
      input.toLowerCase().includes("issue")
      
    if (input.trim().toLowerCase() === "/predict") {
      ws.current.send(
        JSON.stringify({
          type: "property",
          input_data: {
            address: 1308,
            subdistrict_code: 182,
            BEDROOMS: 2.0,
            BATHROOMS: 1.0,
            SIZE: 700.0,
            "PROPERTY TYPE": 9,
            avg_distance_to_nearest_station: 0.4,
            nearest_station_count: 3.0,
          },
        }),
      )
    } else {
      ws.current.send(
        JSON.stringify({
          type: "text",
          message: input,
        }),
      )
    }

    // Simulate showing alerts button for relevant queries
    if (isAlertQuery) {
      setTimeout(() => {
        setIsTyping(false)
        setMessages((msgs) => [
          ...msgs,
          {
            sender: "bot",
            text: `I found ${alerts.length} alerts for your properties. You can view detailed information about maintenance issues, inspections, and urgent matters.`,
            timestamp: new Date(),
            showAlertsButton: true,
          },
        ])
      }, 1500)
    }

    setInput("")
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

      {/* Notification Button */}
      <div className="absolute top-6 right-6 z-50">
        <Button
          onClick={() => setIsAlertsOpen(!isAlertsOpen)}
          variant="outline"
          size="icon"
          className="relative bg-white/90 backdrop-blur-md border-2 border-white/50 shadow-2xl hover:shadow-3xl transition-all duration-300 hover:scale-105"
        >
          <Bell className="h-5 w-5" />
          {urgentAlerts > 0 && (
            <Badge className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center bg-red-500 text-white text-xs animate-pulse">
              {urgentAlerts > 5 ? '5+' : urgentAlerts}
            </Badge>
          )}
        </Button>
      </div>

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
                  {messages.map((msg, idx) => (
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
                          <div
                            className={`inline-block p-4 rounded-2xl shadow-lg ${
                              msg.sender === "user"
                                ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-br-md"
                                : "bg-white/80 backdrop-blur-sm text-gray-800 rounded-bl-md border border-gray-200/50"
                            }`}
                          >
                            {msg.sender === "bot" ? (
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
                            )}
                          </div>
                          {msg.timestamp && isClient && (
                            <div className={`text-xs text-gray-400 mt-1 ${msg.sender === "user" ? "text-right" : ""}`}>
                              {formatTime(msg.timestamp)}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Alerts Button */}
                      {msg.showAlertsButton && (
                        <div className="mt-3 flex justify-start">
                          <Button
                            onClick={() => setIsAlertsOpen(true)}
                            variant="outline"
                            className="bg-blue-50/80 backdrop-blur-sm border-blue-200 text-blue-700 hover:bg-blue-100/80 transition-all duration-200 shadow-md hover:shadow-lg"
                          >
                            <Bell className="h-4 w-4 mr-2" />
                            View All Alerts ({alerts.length})
                            <ChevronRight className="h-4 w-4 ml-2" />
                          </Button>
                        </div>
                      )}
                    </div>
                  ))}

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
              <div className="mt-3 text-center">
                <p className="text-xs text-gray-600">
                  💡 Tip: Ask for any of the three tasks told above from the Assistant!
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Alerts Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full w-96 bg-white/95 backdrop-blur-lg shadow-2xl transform transition-transform duration-300 z-40 ${
          isAlertsOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex flex-col h-full">
          <div className="p-6 border-b bg-gradient-to-r from-red-500 to-orange-500 text-white relative overflow-hidden flex-shrink-0">
            {/* Sidebar header pattern */}
            <div
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `repeating-linear-gradient(
                -45deg,
                transparent,
                transparent 10px,
                rgba(255,255,255,0.1) 10px,
                rgba(255,255,255,0.1) 20px
              )`,
              }}
            ></div>

            <div className="flex items-center justify-between relative z-10">
              <div className="flex items-center space-x-3">
                <Bell className="h-6 w-6" />
                <div>
                  <h2 className="text-xl font-bold">Property Alerts</h2>
                  <p className="text-red-100 text-sm">{alerts.length} active alerts</p>
                </div>
              </div>
              <Button
                onClick={() => setIsAlertsOpen(false)}
                variant="ghost"
                size="icon"
                className="text-white hover:bg-white/20 transition-all duration-200 hover:scale-110"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
          </div>

          <ScrollArea className="flex-1">
            <div className="p-4 space-y-4">
              {alerts.map((alert) => (
                <Card
                  key={alert.id}
                  className="border-l-4 border-l-red-400 shadow-lg hover:shadow-xl transition-all duration-200 bg-white/90 backdrop-blur-sm hover:scale-105"
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getAlertIcon(alert.type)}
                        <span className="font-semibold text-gray-900">[{alert.title}]</span>
                      </div>
                      <Badge className={`text-xs ${getPriorityColor(alert.priority)}`}>
                        {alert.priority.toUpperCase()}
                      </Badge>
                    </div>

                    <div className="space-y-2 text-sm">
                      <div className="font-medium text-blue-600">{alert.property}</div>
                      <div className="flex items-center text-gray-600">
                        <MapPin className="h-3 w-3 mr-1" />
                        {alert.address}
                      </div>

                      <Separator className="my-2" />

                      <div>
                        <span className="font-medium text-gray-700">Recommended Action:</span>
                        <p className="text-gray-600 mt-1">{alert.action}</p>
                      </div>

                      <div>
                        <span className="font-medium text-gray-700">Risk Factors:</span>
                        <p className="text-gray-600 mt-1">{alert.riskFactors}</p>
                      </div>

                      <div className="flex items-center text-gray-500 text-xs">
                        <Clock className="h-3 w-3 mr-1" />
                        Last Inspection: {alert.lastInspection}
                      </div>

                      {isClient && <div className="text-xs text-gray-400">{formatTime(alert.timestamp)}</div>}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>

          <div className="p-4 border-t bg-white/50 backdrop-blur-sm flex-shrink-0">
            <Button className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Manage All Alerts
            </Button>
          </div>
        </div>
      </div>

      {/* Enhanced Overlay */}
      {isAlertsOpen && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-30" onClick={() => setIsAlertsOpen(false)} />
      )}
    </div>
  )
}
