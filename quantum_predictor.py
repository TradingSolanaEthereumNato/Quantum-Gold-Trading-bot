import numpy as np
import logging
import asyncio
import pennylane as qml
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator  # Changed to 'validator' for Pydantic v1.x
from dotenv import load_dotenv
import psutil
from fastapi.responses import JSONResponse
import os
from pennylane import numpy as np

# Load environment variables (API keys, etc.)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Pydantic Models
class MarketData(BaseModel):
    market_data: list[float]

    @validator('market_data')  # Adjusted for Pydantic v1.x
    def check_market_data_length(cls, v):
        if len(v) != 4:
            raise ValueError('market_data must contain exactly 4 elements')
        return v

class QuantumResponse(BaseModel):
    prediction: float
    confidence: float
    quantum_state_hash: Optional[int]

class QuantumError(Exception):
    """Custom exception for quantum-related errors."""
    pass

class QuantumPredictor:
    def __init__(self):
        self.device = self._initialize_quantum_device()
        self.circuit = self._build_quantum_circuit()
        self._validate_circuit()

    def _initialize_quantum_device(self):
        # Try to initialize Rigetti's quantum device first
        try:
            from rigetti import pyquil
            device = pyquil.api.QVMConnection()  # Rigetti's QVM for simulation
            logger.info("Connected to Rigetti quantum cloud device.")
        except ImportError as e:
            logger.warning(f"Rigetti not found, falling back to PennyLane simulator: {str(e)}")
            # Fallback to PennyLane's local simulator if Rigetti isn't available
            device = qml.device("default.qubit", wires=4)
        return device

    def _build_quantum_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params: np.ndarray):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CRY(params[2], wires=[1, 2])
            qml.RX(params[3], wires=3)
            qml.CSWAP(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(2))
        return circuit

    async def predict(self, market_data: np.ndarray) -> dict:
        try:
            processed_data = await self._async_quantum_feature_engineering(market_data)
            results = await self._async_execute_circuit(processed_data)
            return {
                'prediction': np.mean(results),
                'confidence': self._calculate_confidence(results),
                'quantum_state': self._get_quantum_state_safe()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e), 'prediction': None, 'confidence': 0.0}

    async def _async_quantum_feature_engineering(self, data: np.ndarray) -> np.ndarray:
        return await asyncio.to_thread(self._quantum_feature_engineering, data)

    def _quantum_feature_engineering(self, data: np.ndarray) -> np.ndarray:
        max_val = max(data) if max(data) != 0 else 1
        return np.array([[2 * np.pi * (val / max_val) for val in data]])

    async def _async_execute_circuit(self, params: np.ndarray) -> np.ndarray:
        return await asyncio.to_thread(self._execute_circuit_sync, params)

    def _execute_circuit_sync(self, params: np.ndarray) -> np.ndarray:
        return np.array([float(self.circuit(np.array(params)))])

    def _calculate_confidence(self, results: np.ndarray) -> float:
        variance = np.var(results)
        return max(0.0, 1.0 - abs(variance))

    def _get_quantum_state_safe(self) -> Optional[np.ndarray]:
        try:
            return self.device.state if hasattr(self.device, 'state') else None
        except Exception as e:
            return None

    def _validate_circuit(self):
        if not callable(self.circuit):
            raise ValueError("Quantum circuit failed to initialize")
        logger.info("Quantum circuit validated successfully.")
    
    async def _monitor_system_health(self):
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            logger.info(f"System health: CPU usage: {cpu_usage}%")
            await asyncio.sleep(60)

@app.post("/predict", response_model=QuantumResponse)
async def predict(market_data: MarketData):
    predictor = QuantumPredictor()
    asyncio.create_task(predictor._monitor_system_health())
    prediction = await predictor.predict(market_data.market_data)
    if 'error' in prediction:
        raise HTTPException(status_code=500, detail=prediction['error'])
    return prediction

@app.exception_handler(QuantumError)
async def quantum_error_handler(request, exc: QuantumError):
    return JSONResponse(
        status_code=500,
        content={"error": "Quantum computation failed", "details": str(exc)}
    )

# Repository for the project: https://github.com/samskiezz/rigetti-quantum

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

