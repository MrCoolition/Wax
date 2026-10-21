param([string]$OwnerEmail)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$stateDirectory = Join-Path $workspace '.wax-local'
[IO.Directory]::CreateDirectory($stateDirectory) | Out-Null
$statePath = Join-Path $stateDirectory 'setup.json'
function New-WaxSecret {
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    return -join ($bytes | ForEach-Object { $_.ToString('x2') })
}
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
} else {
    $state = @{
        ownerEmail = $OwnerEmail
        ownerToken = New-WaxSecret
        productionAuth = New-WaxSecret
        developmentAuth = New-WaxSecret
        previewAuth = New-WaxSecret
        databases = @(@{ name = 'wax_development'; password = New-WaxSecret }, @{ name = 'wax_preview'; password = New-WaxSecret })
    }
    [IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
}
function Add-WaxEnvironment([string]$Key, [string]$Value, [string]$Environment) {
    $arguments = @('--registry=https://registry.npmjs.org', '--yes', 'vercel@65.0.0', 'env', 'add', $Key, '--environment', $Environment, '--scope', 'coolition', '--force')
    if ($Environment -ne 'development') { $arguments += '--sensitive' }
    $Value | & npx.cmd @arguments
    if ($LASTEXITCODE -ne 0) { throw "Could not configure $Key for $Environment" }
}
Add-WaxEnvironment 'WAX_ENV_PROVISIONING' ($state.databases | ConvertTo-Json -Compress) 'production'
Add-WaxEnvironment 'BETTER_AUTH_SECRET' $state.productionAuth 'production'
Add-WaxEnvironment 'BETTER_AUTH_SECRET' $state.developmentAuth 'development'
Add-WaxEnvironment 'BETTER_AUTH_SECRET' $state.previewAuth 'preview'
Add-WaxEnvironment 'WAX_BOOTSTRAP_TOKEN' $state.ownerToken 'production'
Add-WaxEnvironment 'WAX_OWNER_EMAIL' $state.ownerEmail 'production'
Add-WaxEnvironment 'BETTER_AUTH_URL' 'https://wax-coolition.vercel.app' 'production'
Write-Output 'WAX authentication and isolated-database setup secrets configured. Values were not displayed.'
