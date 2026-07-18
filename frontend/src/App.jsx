import { useEffect, useState } from 'react'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [isPredicting, setIsPredicting] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Backend health check failed')
        }

        return response.json()
      })
      .then((data) => setBackendStatus(data.status))
      .catch(() => setBackendStatus('offline'))
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  function handleImageChange(event) {
    const file = event.target.files?.[0]

    if (!file) {
      setSelectedFile(null)
      setPreviewUrl('')
      setPrediction(null)
      setPredictionError('')
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setPrediction(null)
    setPredictionError('')
  }

  async function handlePredict() {
    if (!selectedFile) {
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    setIsPredicting(true)
    setPrediction(null)
    setPredictionError('')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed')
      }

      setPrediction(data)
    } catch (error) {
      setPredictionError(error.message)
    } finally {
      setIsPredicting(false)
    }
  }

  const statusLabel = {
    checking: 'Checking backend...',
    ok: 'Backend connected',
    offline: 'Backend offline',
  }[backendStatus]

  return (
    <main className="app-shell">
      <h1>AIGOAT Task 2 Depth Estimation</h1>
      <p className={`status status--${backendStatus}`}>{statusLabel}</p>

      <section className="image-picker">
        <label htmlFor="image-upload">Choose a JPG or PNG image</label>
        <input
          id="image-upload"
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleImageChange}
        />

        {selectedFile && (
          <div className="preview">
            <img src={previewUrl} alt="Selected preview" />
            <p>{selectedFile.name}</p>
          </div>
        )}

        <button
          type="button"
          onClick={handlePredict}
          disabled={!selectedFile || isPredicting || backendStatus !== 'ok'}
        >
          {isPredicting ? 'Generating...' : 'Generate depth map'}
        </button>
      </section>

      {predictionError && <p className="error">{predictionError}</p>}

      {prediction && (
        <section className="result">
          <h2>Depth result</h2>
          <img
            src={`data:image/png;base64,${prediction.depth_png_base64}`}
            alt="Predicted depth map"
          />
          <dl>
            <div>
              <dt>Width</dt>
              <dd>{prediction.width}</dd>
            </div>
            <div>
              <dt>Height</dt>
              <dd>{prediction.height}</dd>
            </div>
            <div>
              <dt>Minimum depth</dt>
              <dd>{prediction.min_depth}</dd>
            </div>
            <div>
              <dt>Maximum depth</dt>
              <dd>{prediction.max_depth}</dd>
            </div>
          </dl>
        </section>
      )}
    </main>
  )
}

export default App
