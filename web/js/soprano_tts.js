import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Soprano TTS - Audio player with autoplay functionality
app.registerExtension({
    name: "Soprano.TTS",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SopranoTTS") {
            return;
        }
        
        // Add audio preview widget when node is created
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            // Create audio element for preview
            const audioElement = document.createElement("audio");
            audioElement.controls = true;
            audioElement.style.width = "100%";
            
            // Add the audio widget
            const audioWidget = this.addDOMWidget("audio_preview", "audio", audioElement, {
                serialize: false,
                hideOnZoom: false,
            });
            
            audioWidget.computeSize = function(width) {
                return [width, 54];
            };
            
            // Store reference to audio element on the node
            this.audioElement = audioElement;
            this.audioWidget = audioWidget;
        };
        
        // Handle execution results
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }
            
            if (message && message.audio && message.audio.length > 0) {
                const audioInfo = message.audio[0];
                
                // Build the URL to fetch the audio file
                const params = new URLSearchParams({
                    filename: audioInfo.filename,
                    subfolder: audioInfo.subfolder || "",
                    type: audioInfo.type || "output"
                });
                
                const audioUrl = api.apiURL(`/view?${params.toString()}`);
                
                // Update the audio element source
                if (this.audioElement) {
                    this.audioElement.src = audioUrl;
                    this.audioElement.load();
                    
                    // Autoplay if enabled (flag is in the audio info object)
                    if (audioInfo.autoplay) {
                        this.audioElement.play().catch(err => {
                            console.log("[SopranoTTS] Autoplay blocked by browser:", err);
                        });
                    }
                }
            }
        };
    }
});

