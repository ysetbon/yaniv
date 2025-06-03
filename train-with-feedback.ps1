# Yaniv AI Training Script with Progress Feedback
Clear-Host
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           🎮 YANIV AI TRAINING SYSTEM 🎮                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$episodes = if ($args[0]) { $args[0] } else { 4000 }
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  • Episodes: $episodes" -ForegroundColor White
Write-Host "  • Training every: 40 episodes" -ForegroundColor White
Write-Host "  • Saving every: 100 episodes" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  IMPORTANT: Training on CPU is slow!" -ForegroundColor Red
Write-Host "   Each episode = 1 complete game" -ForegroundColor White
Write-Host "   Expect ~2-3 seconds per episode" -ForegroundColor White
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Run the Node.js training script
& node train-progress-bar.js $episodes