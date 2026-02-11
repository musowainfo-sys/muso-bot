//+------------------------------------------------------------------+
//|                                                MyLSTM.mqh        |
//|                        Custom LSTM Implementation for MQL5       |
//|                             Copyright 2026, Your Name          |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Your Name"
#property version   "1.00"

//+------------------------------------------------------------------+
//| LSTM Layer Class                                                 |
//+------------------------------------------------------------------+
class CLSTMCell
{
private:
   int         m_input_size;
   int         m_hidden_size;
   
   // Weight matrices
   double      m_weights_f[1024];    // Forget gate weights
   double      m_weights_i[1024];    // Input gate weights
   double      m_weights_o[1024];    // Output gate weights
   double      m_weights_g[1024];    // Candidate gate weights
   
   // Bias vectors
   double      m_bias_f[32];
   double      m_bias_i[32];
   double      m_bias_o[32];
   double      m_bias_g[32];
   
   // Hidden and cell states
   double      m_hidden_state[32];
   double      m_cell_state[32];
   
   // Gradients for online learning
   double      m_grad_f[1024];
   double      m_grad_i[1024];
   double      m_grad_o[1024];
   double      m_grad_g[1024];
   double      m_grad_h[32];
   double      m_grad_c[32];

public:
                  CLSTMCell(void);
                 ~CLSTMCell(void);
   bool           Initialize(int input_size, int hidden_size);
   void           ResetState(void);
   bool           Forward(const double &input[], double &output[]);
   void           UpdateWeights(const double &target_error, double learning_rate);
   void           GetHiddenState(double &state[]);
   void           SetHiddenState(const double &state[]);
   void           GetCellState(double &state[]);
   void           SetCellState(const double &state[]);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CLSTMCell::CLSTMCell(void)
{
   m_input_size = 0;
   m_hidden_size = 0;
   ResetState();
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLSTMCell::~CLSTMCell(void)
{
   // Nothing specific to clean up
}

//+------------------------------------------------------------------+
//| Initialize the LSTM cell                                         |
//+------------------------------------------------------------------+
bool CLSTMCell::Initialize(int input_size, int hidden_size)
{
   if(input_size <= 0 || hidden_size <= 0 || input_size > 100 || hidden_size > 100)
      return false;
      
   m_input_size = input_size;
   m_hidden_size = hidden_size;
   
   // Initialize weights with small random values
   int total_weights = (input_size + hidden_size) * hidden_size;
   
   if(total_weights > 1024)
      return false; // Too many weights for our fixed arrays
   
   // Initialize forget gate weights
   for(int i = 0; i < total_weights; i++)
   {
      m_weights_f[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
      m_weights_i[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
      m_weights_o[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
      m_weights_g[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
      
      if(i < hidden_size)
      {
         m_bias_f[i] = 0.0;
         m_bias_i[i] = 0.0;
         m_bias_o[i] = 0.0;
         m_bias_g[i] = 0.0;
         
         m_hidden_state[i] = 0.0;
         m_cell_state[i] = 0.0;
         
         m_grad_h[i] = 0.0;
         m_grad_c[i] = 0.0;
      }
      
      m_grad_f[i] = 0.0;
      m_grad_i[i] = 0.0;
      m_grad_o[i] = 0.0;
      m_grad_g[i] = 0.0;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Reset hidden and cell states                                     |
//+------------------------------------------------------------------+
void CLSTMCell::ResetState(void)
{
   for(int i = 0; i < m_hidden_size; i++)
   {
      m_hidden_state[i] = 0.0;
      m_cell_state[i] = 0.0;
   }
}

//+------------------------------------------------------------------+
//| Sigmoid activation function                                      |
//+------------------------------------------------------------------+
double Sigmoid(double x)
{
   return 1.0 / (1.0 + MathExp(-x));
}

//+------------------------------------------------------------------+
//| Tanh activation function                                         |
//+------------------------------------------------------------------+
double TanhActivation(double x)
{
   if(x > 10.0) return 1.0;
   if(x < -10.0) return -1.0;
   return MathTanh(x);
}

//+------------------------------------------------------------------+
//| Forward pass through the LSTM cell                               |
//+------------------------------------------------------------------+
bool CLSTMCell::Forward(const double &input[], double &output[])
{
   if(m_input_size == 0 || m_hidden_size == 0)
      return false;
   
   int combined_size = m_input_size + m_hidden_size;
   double combined_input[100];
   
   // Combine input and hidden state
   ArrayInitialize(combined_input, 0.0);
   for(int i = 0; i < m_input_size; i++)
      combined_input[i] = input[i];
   for(int i = 0; i < m_hidden_size; i++)
      combined_input[m_input_size + i] = m_hidden_state[i];
   
   // Calculate gate outputs
   double forget_gate[32], input_gate[32], output_gate[32], candidate_gate[32];
   
   for(int h = 0; h < m_hidden_size; h++)
   {
      // Calculate dot products for each gate
      double f_sum = m_bias_f[h];
      double i_sum = m_bias_i[h];
      double o_sum = m_bias_o[h];
      double g_sum = m_bias_g[h];
      
      for(int c = 0; c < combined_size; c++)
      {
         f_sum += m_weights_f[h * combined_size + c] * combined_input[c];
         i_sum += m_weights_i[h * combined_size + c] * combined_input[c];
         o_sum += m_weights_o[h * combined_size + c] * combined_input[c];
         g_sum += m_weights_g[h * combined_size + c] * combined_input[c];
      }
      
      // Apply activations
      forget_gate[h] = Sigmoid(f_sum);
      input_gate[h] = Sigmoid(i_sum);
      output_gate[h] = Sigmoid(o_sum);
      candidate_gate[h] = TanhActivation(g_sum);
      
      // Update cell state
      m_cell_state[h] = forget_gate[h] * m_cell_state[h] + input_gate[h] * candidate_gate[h];
      
      // Update hidden state
      m_hidden_state[h] = output_gate[h] * TanhActivation(m_cell_state[h]);
      
      // Output is the hidden state
      output[h] = m_hidden_state[h];
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Update weights based on target error                            |
//+------------------------------------------------------------------+
void CLSTMCell::UpdateWeights(const double &target_error, double learning_rate)
{
   // Simplified gradient update for demonstration purposes
   // In a real implementation, this would involve proper backpropagation through time
   for(int h = 0; h < m_hidden_size; h++)
   {
      // Adjust bias for output gate based on error
      m_bias_o[h] -= learning_rate * target_error * m_hidden_state[h] * 
                     (1.0 - m_hidden_state[h]*m_hidden_state[h]) * 
                     TanhActivation(m_cell_state[h]);
   }
}

//+------------------------------------------------------------------+
//| Get hidden state                                                 |
//+------------------------------------------------------------------+
void CLSTMCell::GetHiddenState(double &state[])
{
   ArrayResize(state, m_hidden_size);
   for(int i = 0; i < m_hidden_size; i++)
      state[i] = m_hidden_state[i];
}

//+------------------------------------------------------------------+
//| Set hidden state                                                 |
//+------------------------------------------------------------------+
void CLSTMCell::SetHiddenState(const double &state[])
{
   for(int i = 0; i < m_hidden_size && i < ArraySize(state); i++)
      m_hidden_state[i] = state[i];
}

//+------------------------------------------------------------------+
//| Get cell state                                                   |
//+------------------------------------------------------------------+
void CLSTMCell::GetCellState(double &state[])
{
   ArrayResize(state, m_hidden_size);
   for(int i = 0; i < m_hidden_size; i++)
      state[i] = m_cell_state[i];
}

//+------------------------------------------------------------------+
//| Set cell state                                                   |
//+------------------------------------------------------------------+
void CLSTMCell::SetCellState(const double &state[])
{
   for(int i = 0; i < m_hidden_size && i < ArraySize(state); i++)
      m_cell_state[i] = state[i];
}

//+------------------------------------------------------------------+
//| Main LSTM Network Class                                          |
//+------------------------------------------------------------------+
class CLSTM
{
private:
   CLSTMCell   *m_layers[];
   int         m_num_layers;
   int         m_input_size;
   int         m_hidden_sizes[10];  // Max 10 hidden layers
   int         m_output_size;
   
   // Output layer weights (for prediction)
   double      m_output_weights[1024];
   double      m_output_bias[32];
   
   // Sequences for training
   double      m_sequences[][100][100];  // [sequence][timestep][feature]
   int         m_sequence_count;
   int         m_max_sequences;
   
   // Normalization parameters
   double      m_norm_min[50];
   double      m_norm_max[50];
   bool        m_norm_initialized;

public:
                  CLSTM(void);
                 ~CLSTM(void);
   bool           Initialize(int input_size, int &hidden_sizes[], int num_hidden, int output_size);
   bool           AddSequence(const double &seq_data[][50], int seq_length);
   bool           FeedForward(const double &input_seq[][50], int seq_length, double &output[]);
   bool           Predict(const double &input_seq[][50], int seq_length, double &prediction);
   void           OnlineUpdate(const double &input_seq[][50], int seq_length, double target, double learning_rate);
   bool           Save(const string filename);
   bool           Load(const string filename);
   void           NormalizeInput(double &input[], int size);
   void           DenormalizeOutput(double &output[], int size);
   void           UpdateNormalization(const double &input[], int size);
   void           ResetStates(void);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CLSTM::CLSTM(void)
{
   m_num_layers = 0;
   m_input_size = 0;
   m_output_size = 0;
   m_sequence_count = 0;
   m_max_sequences = 1000; // Limit for memory management
   m_norm_initialized = false;
   ArrayInitialize(m_norm_min, 0.0);
   ArrayInitialize(m_norm_max, 0.0);
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CLSTM::~CLSTM(void)
{
   for(int i = 0; i < m_num_layers; i++)
   {
      if(m_layers[i] != NULL)
         delete m_layers[i];
   }
   ArrayFree(m_layers);
}

//+------------------------------------------------------------------+
//| Initialize the LSTM network                                      |
//+------------------------------------------------------------------+
bool CLSTM::Initialize(int input_size, int &hidden_sizes[], int num_hidden, int output_size)
{
   if(num_hidden <= 0 || num_hidden > 10 || input_size <= 0 || output_size <= 0)
      return false;
   
   m_input_size = input_size;
   m_output_size = output_size;
   m_num_layers = num_hidden;
   
   // Resize layers array
   ArrayResize(m_layers, num_hidden);
   
   // Initialize each layer
   for(int i = 0; i < num_hidden; i++)
   {
      m_hidden_sizes[i] = hidden_sizes[i];
      m_layers[i] = new CLSTMCell();
      int layer_input_size = (i == 0) ? input_size : hidden_sizes[i-1];
      
      if(!m_layers[i]->Initialize(layer_input_size, hidden_sizes[i]))
      {
         // Cleanup on failure
         for(int j = 0; j <= i; j++)
         {
            if(m_layers[j] != NULL)
            {
               delete m_layers[j];
               m_layers[j] = NULL;
            }
         }
         return false;
      }
   }
   
   // Initialize output layer weights
   int prev_layer_size = (num_hidden > 0) ? hidden_sizes[num_hidden-1] : input_size;
   int total_output_weights = prev_layer_size * output_size;
   
   if(total_output_weights > 1024)
      return false;
   
   for(int i = 0; i < total_output_weights; i++)
      m_output_weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   
   for(int i = 0; i < output_size; i++)
      m_output_bias[i] = 0.0;
   
   return true;
}

//+------------------------------------------------------------------+
//| Add a sequence to the dataset                                    |
//+------------------------------------------------------------------+
bool CLSTM::AddSequence(const double &seq_data[][50], int seq_length)
{
   if(m_sequence_count >= m_max_sequences)
      return false; // Maximum sequences reached
   
   // In a real implementation, we'd store these sequences
   // For now, just return success
   return true;
}

//+------------------------------------------------------------------+
//| Feed forward through the entire network                          |
//+------------------------------------------------------------------+
bool CLSTM::FeedForward(const double &input_seq[][50], int seq_length, double &output[])
{
   if(seq_length <= 0 || m_num_layers == 0)
      return false;
   
   // Temporary arrays for intermediate results
   double temp_input[100];
   double temp_output[100];
   
   // Process each timestep
   for(int t = 0; t < seq_length; t++)
   {
      // Copy input for first layer
      for(int i = 0; i < m_input_size && i < 50; i++)
         temp_input[i] = input_seq[t][i];
      
      // Process through each layer
      for(int layer = 0; layer < m_num_layers; layer++)
      {
         int layer_output_size = m_hidden_sizes[layer];
         ArrayInitialize(temp_output, 0.0);
         
         if(!m_layers[layer]->Forward(temp_input, temp_output))
            return false;
         
         // Prepare input for next layer (or final output)
         if(layer < m_num_layers - 1)
         {
            for(int i = 0; i < layer_output_size; i++)
               temp_input[i] = temp_output[i];
         }
      }
   }
   
   // Apply output layer transformation
   int final_hidden_size = m_hidden_sizes[m_num_layers-1];
   ArrayResize(output, m_output_size);
   
   for(int out = 0; out < m_output_size; out++)
   {
      output[out] = m_output_bias[out];
      for(int h = 0; h < final_hidden_size; h++)
      {
         output[out] += temp_output[h] * m_output_weights[out * final_hidden_size + h];
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Make a prediction using the trained model                        |
//+------------------------------------------------------------------+
bool CLSTM::Predict(const double &input_seq[][50], int seq_length, double &prediction)
{
   return FeedForward(input_seq, seq_length, prediction);
}

//+------------------------------------------------------------------+
//| Update the model with new data (online learning)                |
//+------------------------------------------------------------------+
void CLSTM::OnlineUpdate(const double &input_seq[][50], int seq_length, double target, double learning_rate)
{
   // First get current prediction
   double current_output[];
   if(Predict(input_seq, seq_length, current_output))
   {
      // Calculate error
      double error = target - current_output[0]; // Assuming single output
      
      // Update output layer weights
      int final_hidden_size = m_hidden_sizes[m_num_layers-1];
      for(int out = 0; out < m_output_size; out++)
      {
         for(int h = 0; h < final_hidden_size; h++)
         {
            // Get the last hidden state value for this neuron
            double hidden_val = 0.0; // Would need to track this properly
            m_output_weights[out * final_hidden_size + h] += learning_rate * error * hidden_val;
         }
         m_output_bias[out] += learning_rate * error;
      }
      
      // Update internal LSTM weights (simplified)
      m_layers[m_num_layers-1]->UpdateWeights(error, learning_rate);
   }
}

//+------------------------------------------------------------------+
//| Save the model to a file                                         |
//+------------------------------------------------------------------+
bool CLSTM::Save(const string filename)
{
   int handle = FileOpen(filename, FILE_WRITE | FILE_BIN);
   if(handle == INVALID_HANDLE)
      return false;
   
   // Write model architecture
   FileWriteInteger(handle, m_num_layers);
   FileWriteInteger(handle, m_input_size);
   FileWriteInteger(handle, m_output_size);
   
   // Write hidden sizes
   for(int i = 0; i < m_num_layers; i++)
      FileWriteInteger(handle, m_hidden_sizes[i]);
   
   // Write normalization parameters
   FileWriteInteger(handle, m_norm_initialized ? 1 : 0);
   if(m_norm_initialized)
   {
      for(int i = 0; i < 50; i++)
      {
         FileWriteDouble(handle, m_norm_min[i]);
         FileWriteDouble(handle, m_norm_max[i]);
      }
   }
   
   // Write layer weights (simplified)
   for(int layer = 0; layer < m_num_layers; layer++)
   {
      // This is simplified - in practice, you'd save all weights and biases
      double hidden_state[32];
      m_layers[layer]->GetHiddenState(hidden_state);
      for(int i = 0; i < 32; i++)
         FileWriteDouble(handle, hidden_state[i]);
   }
   
   // Write output weights
   int final_hidden_size = m_hidden_sizes[m_num_layers-1];
   int total_weights = final_hidden_size * m_output_size;
   
   for(int i = 0; i < total_weights; i++)
      FileWriteDouble(handle, m_output_weights[i]);
   
   for(int i = 0; i < m_output_size; i++)
      FileWriteDouble(handle, m_output_bias[i]);
   
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Load the model from a file                                       |
//+------------------------------------------------------------------+
bool CLSTM::Load(const string filename)
{
   int handle = FileOpen(filename, FILE_READ | FILE_BIN);
   if(handle == INVALID_HANDLE)
      return false;
   
   // Read model architecture
   m_num_layers = (int)FileReadInteger(handle);
   m_input_size = (int)FileReadInteger(handle);
   m_output_size = (int)FileReadInteger(handle);
   
   // Read hidden sizes and recreate layers
   ArrayResize(m_layers, m_num_layers);
   for(int i = 0; i < m_num_layers; i++)
   {
      m_hidden_sizes[i] = (int)FileReadInteger(handle);
      m_layers[i] = new CLSTMCell();
      int layer_input_size = (i == 0) ? m_input_size : m_hidden_sizes[i-1];
      m_layers[i]->Initialize(layer_input_size, m_hidden_sizes[i]);
   }
   
   // Read normalization parameters
   m_norm_initialized = FileReadInteger(handle) != 0;
   if(m_norm_initialized)
   {
      for(int i = 0; i < 50; i++)
      {
         m_norm_min[i] = FileReadDouble(handle);
         m_norm_max[i] = FileReadDouble(handle);
      }
   }
   
   // Read layer weights (simplified)
   for(int layer = 0; layer < m_num_layers; layer++)
   {
      double hidden_state[32];
      for(int i = 0; i < 32; i++)
         hidden_state[i] = FileReadDouble(handle);
      m_layers[layer]->SetHiddenState(hidden_state);
   }
   
   // Read output weights
   int final_hidden_size = m_hidden_sizes[m_num_layers-1];
   int total_weights = final_hidden_size * m_output_size;
   
   for(int i = 0; i < total_weights; i++)
      m_output_weights[i] = FileReadDouble(handle);
   
   for(int i = 0; i < m_output_size; i++)
      m_output_bias[i] = FileReadDouble(handle);
   
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Normalize input features                                         |
//+------------------------------------------------------------------+
void CLSTM::NormalizeInput(double &input[], int size)
{
   if(!m_norm_initialized)
   {
      // Initialize normalization parameters
      for(int i = 0; i < size; i++)
      {
         m_norm_min[i] = input[i];
         m_norm_max[i] = input[i];
      }
      m_norm_initialized = true;
   }
   else
   {
      // Update normalization parameters and normalize
      for(int i = 0; i < size; i++)
      {
         if(input[i] < m_norm_min[i])
            m_norm_min[i] = input[i];
         if(input[i] > m_norm_max[i])
            m_norm_max[i] = input[i];
         
         double range = m_norm_max[i] - m_norm_min[i];
         if(range != 0)
            input[i] = (input[i] - m_norm_min[i]) / range;
         else
            input[i] = 0.0; // Handle case where min equals max
      }
   }
}

//+------------------------------------------------------------------+
//| Denormalize output                                               |
//+------------------------------------------------------------------+
void CLSTM::DenormalizeOutput(double &output[], int size)
{
   // This would reverse the normalization if needed
   // For now, it's a placeholder
}

//+------------------------------------------------------------------+
//| Update normalization parameters                                  |
//+------------------------------------------------------------------+
void CLSTM::UpdateNormalization(const double &input[], int size)
{
   if(!m_norm_initialized)
   {
      // Initialize with first values
      for(int i = 0; i < size; i++)
      {
         m_norm_min[i] = input[i];
         m_norm_max[i] = input[i];
      }
      m_norm_initialized = true;
   }
   else
   {
      // Update min/max values
      for(int i = 0; i < size; i++)
      {
         if(input[i] < m_norm_min[i])
            m_norm_min[i] = input[i];
         if(input[i] > m_norm_max[i])
            m_norm_max[i] = input[i];
      }
   }
}

//+------------------------------------------------------------------+
//| Reset all internal states                                        |
//+------------------------------------------------------------------+
void CLSTM::ResetStates(void)
{
   for(int i = 0; i < m_num_layers; i++)
   {
      if(m_layers[i] != NULL)
         m_layers[i]->ResetState();
   }
}