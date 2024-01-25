#include<bits/stdc++.h>
using namespace std;

int n;
long long INF = 10*10*10*10*10*10*10*10*10 + 7;
int arr[1000000];
long long ketqua = 0;

void Try(){
    for(int i=0; i<n-1; i++){
        for(int j=i+1; j<n; j++){
            
            if(arr[i] == arr[j]){
                ketqua++;
            }
        }
    }
}
int main(){

    cin >> n;
    for(int i=0; i<n; i++){
        cin >> arr[i];
    }

    Try();

    int y = ketqua / INF;
    int y1 = ketqua - y*INF;

    cout << y1;
}