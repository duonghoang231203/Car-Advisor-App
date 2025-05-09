import 'dart:convert';
import 'package:http/http.dart' as http;

void main() async {
  // Test the API call with string IDs
  await testCompareWithStringIds();
  
  // Test the API call with integer IDs
  await testCompareWithIntIds();
}

Future<void> testCompareWithStringIds() async {
  print('Testing with string IDs...');
  
  final url = Uri.parse('http://localhost:8000/api/cars/compare');
  final headers = {'Content-Type': 'application/json'};
  
  // Create payload with string IDs
  final payload = {'car_ids': ['1', '2']};
  
  try {
    final response = await http.post(
      url,
      headers: headers,
      body: jsonEncode(payload),
    );
    
    print('Status code: ${response.statusCode}');
    print('Response body: ${response.body}');
  } catch (e) {
    print('Error: $e');
  }
}

Future<void> testCompareWithIntIds() async {
  print('\nTesting with integer IDs...');
  
  final url = Uri.parse('http://localhost:8000/api/cars/compare');
  final headers = {'Content-Type': 'application/json'};
  
  // Create payload with integer IDs
  final payload = {'car_ids': [1, 2]};
  
  try {
    final response = await http.post(
      url,
      headers: headers,
      body: jsonEncode(payload),
    );
    
    print('Status code: ${response.statusCode}');
    print('Response body: ${response.body}');
  } catch (e) {
    print('Error: $e');
  }
}
