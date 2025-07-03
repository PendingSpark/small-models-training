
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "     PowerShell Profile Loaded!"                    -ForegroundColor Green
Write-Host "     User: $env:UserName"                           -ForegroundColor Yellow
Write-Host "     Computer: $env:ComputerName"                   -ForegroundColor Magenta
Write-Host "     Edit this script with: code `$PROFILE"        -ForegroundColor Gray
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

##
## The docker images and versions to be setup in a friendly container name.
## More containers for different use cases can be found here:
##  https://catalog.ngc.nvidia.com/?filters=resourceType%7CContainer%7Ccontainer%2Cplatform%7CPyTorch%7Cpltfm_pytorch%2Cplatform%7CTensorFlow%7Cpltfm_tensorflow&orderBy=scoreDESC&query=&page=&pageSize=
##
## The core arguments to launch the container settings are descibed below
##
$PytorchFriendlyContainerName = 'pytorch-25.06'
$PytorchImage = 'nvcr.io/nvidia/pytorch:25.06-py3'
$PytorchSpecificArgs = @(
    '--gpus', 'all', # NVIDIA Container Toolkit allows us to specify which GPUs. it only recognized nvidia gpus so the intel gpu is left out
    '-i', '-t', # Starts internactive mode to keep docker alive between commands and TTY mode gives us a better terminal interface
    '--shm-size=1g', # Sets the size of the /dev/shm (shared memory) directory inside the container.
    '--ulimit', 'memlock=-1', # some training uses memory locks, -1 sets RLIMIT_MEMLOCK to unlimited. 
    '--ulimit', 'stack=67108864' # size in bytes (64MB) of the stack RLIMIT_STACK - this is a fairly standard large stack size
)

$CustomPytorchFriendlyContainerName = 'custom-pytorch-25.06'
$CustomPytorchImage = 'my-custom-pytorch:latest'
$CustomPytorchSpecificArgs = @(
    '--gpus', 'all', # NVIDIA Container Toolkit allows us to specify which GPUs. it only recognized nvidia gpus so the intel gpu is left out
    '-i', '-t', # Starts internactive mode to keep docker alive between commands and TTY mode gives us a better terminal interface
    '--shm-size=1g', # Sets the size of the /dev/shm (shared memory) directory inside the container.
    '--ulimit', 'memlock=-1', # some training uses memory locks, -1 sets RLIMIT_MEMLOCK to unlimited. 
    '--ulimit', 'stack=67108864' # size in bytes (64MB) of the stack RLIMIT_STACK - this is a fairly standard large stack size
)

$TensorFlowFriendlyContainerName = 'tensorflow-25.02'
$TensorFlowImage = 'nvcr.io/nvidia/tensorflow:25.02-tf2-py3'
$TensorFlowSpecificArgs = @(
    '--gpus', 'all', # NVIDIA Container Toolkit allows us to specify which GPUs. it only recognized nvidia gpus so the intel gpu is left out
    '-i', '-t', # Starts internactive mode to keep docker alive between commands and TTY mode gives us a better terminal interface
    '--shm-size=1g', # Sets the size of the /dev/shm (shared memory) directory inside the container.
    '--ulimit', 'memlock=-1', # some training uses memory locks, -1 sets RLIMIT_MEMLOCK to unlimited. 
    '--ulimit', 'stack=67108864' # size in bytes (64MB) of the stack RLIMIT_STACK - this is a fairly standard large stack size
)

##
## Helper functions to use the smart container starting for different docker containers
##
function Start-Pytorch {
    Start-DockerContainerSmart `
        -Image $PytorchImage `
        -Name $PytorchFriendlyContainerName `
        -AdditionalArgs $PytorchSpecificArgs
}

function Start-CustomPytorch {
    Start-DockerContainerSmart `
        -Image $CustomPytorchImage `
        -Name $CustomPytorchFriendlyContainerName `
        -AdditionalArgs $CustomPytorchSpecificArgs
}

function Start-TensorFlow {
    Start-DockerContainerSmart `
        -Image $TensorFlowImage `
        -Name $TensorFlowFriendlyContainerName `
        -AdditionalArgs $TensorFlowSpecificArgs
}



##
## Shortcut alias for the PyTorch and TensorFlow to easily start docker containers
##
Set-Alias -Name run-pytorch -Value Start-Pytorch
Set-Alias -Name run-custompytorch -Value Start-CustomPytorch
Set-Alias -Name run-tensorflow -Value Start-TensorFlow


##
## Primary Helper function that creates or starts the docker container (if already created)
##
function Start-DockerContainerSmart {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Image, # The Docker image name (e.g., 'nginx:latest')
        [Parameter(Mandatory=$true)]
        [string]$Name, # The desired container name (e.g., 'my-web-server')
        [string[]]$AdditionalArgs = @() # Any other 'docker run' arguments (e.g., '--gpus all', '-p 80:80')
    )

    Write-Host "Managing Docker container '$Name'..." -ForegroundColor Green

    # Construct the full 'docker run' arguments array
    # This array will be passed to 'docker'
    $runCommandArgs = @('run', '--name', $Name) + $AdditionalArgs + @($Image)

    # 1. Check if a container with this name already exists (any status)
    # Use 2>$null to suppress Docker's error output if the command fails (e.g., if Docker isn't running)
    $existingContainerId = $(docker ps -a --filter "name=$Name" --format "{{.ID}}" 2>$null)

    if ($existingContainerId) {
        # Container exists, now get its status
        $status = $(docker inspect -f '{{.State.Status}}' $Name 2>$null)

        switch ($status) {
            'running' {
                Write-Host "Container '$Name' is already running. Attaching to it..." -ForegroundColor Yellow
                docker attach $Name
            }
            'exited' {
                Write-Host "Container '$Name' exists but is exited. Starting and attaching..." -ForegroundColorYellow
                docker start $Name
                docker attach $Name
            }
            default {
                Write-Warning "Container '$Name' is in an unexpected state ('$status'). Attempting to remove and recreate."
                # Force remove in case it's stuck or in a weird state
                docker rm -f $Name 2>$null | Out-Null # Also suppress errors for rm if container isn't there
                Write-Host "Creating and running new container '$Name'..." -ForegroundColor Green
                & docker @runCommandArgs
            }
        }
    } else {
        # Container does not exist, create and run it for the first time
        Write-Host "Container '$Name' does not exist. Creating and running..." -ForegroundColor Green
        & docker @runCommandArgs
    }
}





# Helptext for available commands (Updated to reflect new structure)
Write-Host ""
Write-Host "Docker Aliases Available:"
Write-Host ""
Write-Host "  To start PyTorch training container:  run-pytorch"
Write-Host "    (Manages container '$PytorchFriendlyContainerName' from image '$PytorchImage')" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  To start TensorFlow training container:  run-tensorflow"
Write-Host "    (Manages container '$TensorFlowFriendlyContainerName' from image '$TensorFlowImage')" -ForegroundColor DarkGray
Write-Host ""
